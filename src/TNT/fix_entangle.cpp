// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_entangle.h"

#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "respa.h"
#include "update.h"

#include <cstring>
#include "math_const.h"
#include "random_mars.h"
#include "string.h"
#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixEntangle::FixEntangle(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), nu(nullptr), N_rest(nullptr), N_0(nullptr), random(nullptr)
{

  // IS THE NUMBER OF ARGUMENTS CORRECT? //
  if (narg < 2) error->all(FLERR,"Illegal fix entangle command");

  // WAS NEVERY SET CORRECTLY? //
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix entangle command");

  zeta = utils::numeric(FLERR,arg[4],false,lmp);
  if (zeta <= 0) error->all(FLERR,"Illegal fix entangle command - zeta should be greater than 0");

  n_critical = utils::numeric(FLERR,arg[5],false,lmp);
  if (n_critical <= 0) error->all(FLERR,"Illegal fix entangle command - critical segment count should be greater than 0");

  b = utils::numeric(FLERR,arg[6],false,lmp);
  if (b <= 0) error->all(FLERR,"Illegal fix entangle command - Kuhn length should be greater than 0");

  // SOME FLAGS... worry about later //
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  // maybe used later
  dynamic_group_allow = 1;

  // for passing virial contributions
  virial_global_flag = 1;
  virial_peratom_flag = 1;
  thermo_virial = 1;

  // for having a peratom array related to this fix
  peratom_flag = 1;
  size_peratom_cols = 10;
  
  // used later to setup
  countflag = 0;

  // setting a seed for random number generation
  seed = 12345;

  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + me);

  // INITIALIZE ANY LOCAL ARRAYS //
  nu = nullptr;
  N_rest = nullptr;
  N_0 = nullptr;

  nmax = atom->nmax;

  // set up reneighboring
  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  dis_flag = 0;

  // for allocating space to array_atom
  FixEntangle::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);

  if (peratom_flag) {
    init_myarray();
  }

  // Set forward communication size
  comm_forward = 1 + atom->maxspecial;
}

/* ---------------------------------------------------------------------- */

FixEntangle::~FixEntangle()
{
  // delete random number
  delete random;

  // delete locally stored arrays
  memory->destroy(nu);
  memory->destroy(N_0);
  memory->destroy(N_rest);
  // DELETE CALL TO FIX PROPERTY/ATOM //
  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete [] new_fix_id;

}

/* ---------------------------------------------------------------------- */

int FixEntangle::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEntangle::post_constructor()
{

  // CREATE A CALL TO FIX PROPERTY/ATOM //
  new_fix_id = utils::strdup(id + std::string("_FIX_PA"));
  modify->add_fix(fmt::format("{} {} property/atom d2_nvar_{} {} ghost yes",new_fix_id, group->names[igroup],id,std::to_string(10)));

  // RETURN THE INDEX OF OUR LOCALLY STORED ATOM ARRAY //
  int tmp1, tmp2;
  index = atom->find_custom(utils::strdup(std::string("nvar_")+id),tmp1,tmp2);

  // Create memory allocations
  nmax = atom->nmax;
  memory->create(nu,nmax,2,"entangle:nu");
  
}

/* ---------------------------------------------------------------------- */

void FixEntangle::init()
{

  // Only run this in the beginning of simulation (transferring nvar data from velocities to actual peratom array)
  if (countflag) return;
  countflag = 1;

  // nvar IS THE POINTER TO OUR STATE VARIABLE ARRAY! //
  double **nvar = atom->darray[index];
  
  // "printcounter" is later used for printing custom messages every Nth timestep
  int printcounter = 1;

  // accessing atom count data
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  
  int *mask = atom->mask;

  // BOND INFORMATION
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;

  // accessing per-atom velocities and radius because the contain nvar (from data file)
  double **v = atom->v;
  double *radius = atom->radius;


  // ! INITIALIZE OUR NEW STATE VARIABLE ! //
  // nvar = [ LHS n | RHS n | LHS tag | RHS tag | identified tag1 | identified tag2 ]

  for (int i = 0; i < nall; i++) {
    for (int m = 0; m < 6; m++) {
      if (mask[i] & groupbit) { 
        nvar[i][m] = 0;
      }
    }
  }

  // importing per-atom nvar values from per-atom velocities and radius in data file (Velocities are set back to zero at end of each loop iteration)
  for (int i = 0; i < nlocal; i++) {
    nvar[i][0] = v[i][0]; 
    nvar[i][1] = v[i][1];
    nvar[i][2] = v[i][2];
    if (radius[i] == 0){ // this is because we can't assign -1 to radius in dat file so we convert zeros to -1 as dangling end flags
      nvar[i][3] = -1;
    } else {
      nvar[i][3] = radius[i]*2;
    }

    v[i][0] = 0;
    v[i][1] = 0;
    v[i][2] = 0;
  }

  for (int i = 0; i < nlocal; i++){
    if (nvar[i][2] == -1) nvar[i][0] = 200;
    if (nvar[i][3] == -1) nvar[i][1] = 200;

    if (nvar[i][2] != -1) nvar[i][0] = 200;
    if (nvar[i][3] != -1) nvar[i][1] = 200;
  }

  // forward communicate of nvar so ghost atoms acquire their nvar
  commflag = 1;
  comm->forward_comm(this,10);

}

/* ---------------------------------------------------------------------- */

void FixEntangle::setup(int vflag)
{
  pre_force(vflag);
}

/* ----------------------------------------------------------------------
  perform disentanglement
  done before exchange, borders, reneighbor
  so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */
void FixEntangle::pre_exchange(){

  // if (update->ntimestep > 100000) return;
  return;

  
  // local atom count
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // global tags
  tagint *tag = atom->tag;

  // Molecule tags
  tagint *molecule = atom->molecule;

  // per-atom state variable
  double **nvar = atom->darray[index];

  // BOND INFORMATION
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;

  // bond list from neighbor class
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  // atom positions
  double **x = atom->x;

  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();

  // if (0){

  // entanglement creation part (IT IS GONNA BE DONE THIS TIME)

  for (int i = 0; i < nlocal; i++){
    // break;
    // this is the cut-off for finding potential neighbors (bonds with head particles farther than this will not be considered)
    double reach_radius = 0;
    double dang_end_monomer = 0;

    // maximum reach radius for a dangling end is equal to it's Contour length (b * n)
    if (nvar[i][2] == -1) {
      reach_radius = b * nvar[i][0];
      dang_end_monomer = nvar[i][0];
    }

    if (nvar[i][3] == -1) {
      reach_radius = b * nvar[i][1];
      dang_end_monomer = nvar[i][1];
    }

    if (reach_radius == 0) continue; // continue to next particle if particle "i" is not a dangling end

    if (dang_end_monomer < 3000) continue;

    // printf("\n\nDANG END MONOMER OF PARTICLE %d = %f\n\n",tag[i],dang_end_monomer);

    double chosen_segment[3] = {0 , 0 , 0};

    // loop over bondlist to find bonds that have their heads inside reach radius of particle "i"

    for (int j = 0; j < nbondlist; j++){
      int i1 = bondlist[j][0];
      int i2_tmp = bondlist[j][1];
      int type = bondlist[j][2];

      if (type == 2) continue;  // do not consider type 2 bonds (entanglement bonds)
      if (i1 == i || i2_tmp == i) continue; // do not consider the previous segment to that dangling end

      int i2 = domain->closest_image(i1,i2_tmp);  

      if (((nvar[i1][2] == tag[i2] || nvar[i1][3] == tag[i2]) && (nvar[i2][2] == tag[i1] || nvar[i2][3] == tag[i1]))== 0){
        continue;
      }

      double x_midpoint = (x[i1][0] + x[i2][0])/2;
      double y_midpoint = (x[i1][1] + x[i2][1])/2;
      double z_midpoint = (x[i1][2] + x[i2][2])/2;

      double dist_x1 = x_midpoint - x[i][0];
      double dist_y1 = y_midpoint - x[i][1];
      double dist_z1 = z_midpoint - x[i][2];
      domain->minimum_image(dist_x1, dist_y1, dist_z1);

      double distance = sqrt((dist_x1) * (dist_x1) + (dist_y1) * (dist_y1) + (dist_z1) * (dist_z1));

      if (distance >= reach_radius) continue;  // do not consider segments with midpoints farther than reach radius

      if (distance >= 8) continue;

      if (chosen_segment[2] == 0){  // if no previous segment has been saved, store this one anyway
        chosen_segment[0] = tag[i1];
        chosen_segment[1] = tag[i2];
        chosen_segment[2] = distance;

      } else if (distance < chosen_segment[2]){ // if the distance is smaller than the distance to previously stored segment, this segment is more probable so replace it
        chosen_segment[0] = tag[i1];
        chosen_segment[1] = tag[i2];
        chosen_segment[2] = distance;

      }

    }
    // continue to next dangling end if no segment has been chosen
    if (chosen_segment[0] == 0 || chosen_segment[1] == 0) continue;

    double rest_length = b * pow(dang_end_monomer,0.6);
    double tau_0 = 50000000;
    double Tau_ent = 0;

    if (chosen_segment[2] < rest_length){   // if potential segment's midpoint is closer than rest length
      Tau_ent = tau_0;
    } else if (chosen_segment[2] >= rest_length){   // if potential segment's midpoint is farther than rest length (needs to diffuse there)
      Tau_ent = dang_end_monomer * (chosen_segment[2] - rest_length) * (chosen_segment[2] - rest_length) / (chosen_segment[2] - dang_end_monomer*b) + tau_0;
    }

    // rate of entanglement creation
    double K_ent = 1 / Tau_ent;

    // calculate probability of entanglement formation for this K_ent
    double P = 1 - exp(-K_ent * update->ntimestep);
    double random_thresh = random->uniform();

    if (P > random_thresh){
      nvar[i][4] = chosen_segment[0];
      nvar[i][5] = chosen_segment[1];

      int id1 = atom->map(chosen_segment[0]);
      int id2 = atom->map(chosen_segment[1]);

      double x_midpoint = (x[id1][0] + x[id2][0])/2;
      double y_midpoint = (x[id1][1] + x[id2][1])/2;
      double z_midpoint = (x[id1][2] + x[id2][2])/2;

      nvar[i][6] = x_midpoint;
      nvar[i][7] = y_midpoint;
      nvar[i][8] = z_midpoint;
    }
  }

  commflag = 1;
  comm->forward_comm(this,10);


  // finding current max atom IDs (between all processors)
  find_maxid();

  // clear ghost count (and atom map) and any ghost bonus data
  // do it now b/c inserting atoms will overwrite ghost atoms

  if (atom->map_style != Atom::MAP_NONE) atom->map_clear();
  atom->nghost = 0;
  atom->avec->clear_bonus();

  int success = 0;

  // creating the particle pair (Only create the new particle pair if the location is in owned region)
  for (int i = 0; i < nall; i++){
    
    if (nvar[i][4] != 0 && nvar[i][5] != 0) {   // finding the anchorpoints which must create new entanglements
      
      int Anchorpoint = i;
      if (nvar[Anchorpoint][2] != -1 && nvar[Anchorpoint][3] != -1) error->one(FLERR,"Wrong entanglement creation has been identified");
      
      // now we can calculate the position of particles we want to deposit

      double x_mid = nvar[i][6];
      double y_mid = nvar[i][7];
      double z_mid = nvar[i][8];

      double Xmid[3] = {x_mid, y_mid , z_mid};

      domain->remap(Xmid);

      x_mid = Xmid[0];
      y_mid = Xmid[1];
      z_mid = Xmid[2];
      

      // accessing processor domain boundaries to check if position is inside domain
      double *sublo,*subhi;
      if (domain->triclinic == 0) {
        sublo = domain->sublo;
        subhi = domain->subhi;
      }
      
      int IN_flag = 0;

      if (x_mid > sublo[0] && x_mid < subhi[0] &&
        y_mid > sublo[1] && y_mid < subhi[1] &&
        z_mid > sublo[2] && z_mid < subhi[2]) IN_flag = 1;


      if (IN_flag == 0) continue; // do not create particle pair if the position is out of proc boundaries
      
      int IN_flag_ent1 = 0;
      int IN_flag_ent2 = 0;

      double x_ent1[3] = {0.0 , 0.0 , 0.0};
      double x_ent2[3] = {0.0 , 0.0 , 0.0};

      // iterate until both x_ent1 and x_ent2 are in processor domain
      while (IN_flag_ent1 == 0 || IN_flag_ent2 == 0){
        if (IN_flag_ent1 == 0){
          double rand1 = random->uniform();

          x_ent1[0] = x_mid + 0.1 * rand1;
          x_ent1[1] = y_mid + 0.1 * sqrt(1 - rand1 * rand1);
          x_ent1[2] = z_mid;

          domain->remap(x_ent1);

          if (x_ent1[0] > sublo[0] && x_ent1[0] < subhi[0] &&
            x_ent1[1] > sublo[1] && x_ent1[1] < subhi[1] &&
            x_ent1[2] > sublo[2] && x_ent1[2] < subhi[2]) IN_flag_ent1 = 1;
        }

        if (IN_flag_ent2 == 0){
          double rand2 = random->uniform();

          x_ent2[0] = x_mid + 0.1 * rand2;
          x_ent2[1] = y_mid + 0.1 * sqrt(1 - rand2 * rand2);
          x_ent2[2] = z_mid;

          domain->remap(x_ent2);

          if (x_ent2[0] > sublo[0] && x_ent2[0] < subhi[0] &&
            x_ent2[1] > sublo[1] && x_ent2[1] < subhi[1] &&
            x_ent2[2] > sublo[2] && x_ent2[2] < subhi[2]) IN_flag_ent2 = 1;
        }
      }

      printf("\n\nnlocal of proc %d (before) = %d at timestep %ld\n\n",me,atom->nlocal,update->ntimestep);
      success = 1;

      int ii = -1;
      ii = atom->nlocal + atom->nghost;
      if (ii >= atom->nmax) atom->avec->grow(0);
      
      // Now that we know the positions of particle pair we can deposit them
      atom->avec->create_atom(1,x_ent1);
      int ent1_ID = atom->nlocal - 1;   // local ID of newly created atom
      
      atom->mask[ent1_ID] = 1 | groupbit;
      atom->image[ent1_ID] = ((imageint) IMGMAX << IMG2BITS) | ((imageint) IMGMAX << IMGBITS) | IMGMAX;

      atom->v[ent1_ID][0] = 0.0;
      atom->v[ent1_ID][1] = 0.0;
      atom->v[ent1_ID][2] = 0.0;

      modify->create_attribute(ent1_ID);

      nvar[ent1_ID][0] = 0.0;
      nvar[ent1_ID][1] = 0.0;
      if (nvar[Anchorpoint][2] == -1){
        nvar[ent1_ID][3] = tag[Anchorpoint];
        nvar[ent1_ID][2] = -1;
      } else if (nvar[Anchorpoint][3] == -1){
        nvar[ent1_ID][2] = tag[Anchorpoint];
        nvar[ent1_ID][3] = -1;
      }

      // creation of second atom

      ii = atom->nlocal + atom->nghost;
      if (ii >= atom->nmax) atom->avec->grow(0);

      atom->avec->create_atom(1,x_ent2);
      int ent2_ID = atom->nlocal - 1;   // local ID of newly created atom

      atom->mask[ent2_ID] = 1 | groupbit;
      atom->image[ent2_ID] = ((imageint) IMGMAX << IMG2BITS) | ((imageint) IMGMAX << IMGBITS) | IMGMAX;

      atom->v[ent2_ID][0] = 0.0;
      atom->v[ent2_ID][1] = 0.0;
      atom->v[ent2_ID][2] = 0.0;

      modify->create_attribute(ent2_ID);

      nvar[ent2_ID][0] = 0.0;
      nvar[ent2_ID][1] = 0.0;
      nvar[ent2_ID][2] = nvar[Anchorpoint][4];
      nvar[ent2_ID][3] = nvar[Anchorpoint][5];


      atom->natoms += 2;
      maxtag_all += 2;
      printf("\n\nnlocal of proc %d (after) = %d at timestep %ld\n\n",me,atom->nlocal,update->ntimestep);
      break;
    }
  }

  MPI_Barrier(world);

  if (atom->tag_enable) atom->tag_extend();
  atom->tag_check();

  if (update->ntimestep > 6143) {
    //printf("\n\n   atoms just created (PROC %d and TIMESTEP %ld) : %d & %d \n\n",me,update->ntimestep,tag[nlocal-1],tag[nlocal-2]);
  }

  // rebuild atom map
  if (atom->map_style != Atom::MAP_NONE) {
    atom->map_init();
    atom->map_set();
  }

  // domain->pbc();
  // comm->exchange();
  atom->nghost = 0;
  comm->borders();


  // rebuild atom map
  if (atom->map_style != Atom::MAP_NONE) {
    atom->map_init();
    atom->map_set();
  }


  // update nlocal value

  nlocal = atom->nlocal;
  nghost = atom->nghost;
  nall = nlocal + nghost;

  commflag = 1;
  comm->forward_comm(this,10);



  if (update->ntimestep > 1630){
    //printf("\n\n        !!!!       PROCESSOR %d HASTAM ba %d ATOM OWNEDN AND %d GHOST   !!!!          \n\n",me,nlocal,nghost);
    //printf("\n\nPROCESSOR %d with nlocal %d -- atom %d with tag %d with nvar = [%f %f %f %f ...]\n",me,nlocal,800,tag[800],nvar[800][0],nvar[800][1],nvar[800][2],nvar[800][3]);
  }

  // Loop to find newly created pairs and create bond "type 2" between them

  for (int j = 0; j < nlocal; j++){
    
    tagint anchor_point_tag = 0;
    int new_ent1 = -1;
    int new_ent2 = -1;

    int foundflag = 0;

    // first we should identify the particles with "molecule id = 0" (Just created)
    if (molecule[j] == 0){

      if (nvar[j][2] == -1 || nvar[j][3] == -1){

        new_ent1 = j;
        // printf("\nNEWLY CREATED IS PARTICLE %d (tag %d) (PROC %d || timestep : %ld)\n",new_ent1,tag[new_ent1],me,update->ntimestep);
        // printf("\nNVAR OF PARTICLE %d = [%f  %f  %f  %f]\n",tag[new_ent1],nvar[new_ent1][0],nvar[new_ent1][1],nvar[new_ent1][2],nvar[new_ent1][3]);
        // printf("\nPosition of particle %d = [%f %f %f]\n",tag[new_ent1],atom->x[new_ent1][0],atom->x[new_ent1][1],atom->x[new_ent1][2]);
        if (nvar[new_ent1][2] == -1){
          anchor_point_tag = nvar[new_ent1][3];
          foundflag = 1;
        } else if (nvar[new_ent1][3] == -1){
          anchor_point_tag = nvar[new_ent1][2];
          foundflag = 1;
        }
        if (anchor_point_tag == 0) error->one(FLERR,"Inconsistent particle is created / nvar is not right for new particle");
      } else {
        continue;
      }

    } else {
      continue;
    }

    if (foundflag != 1) continue;

    // printf("\nAnchor point tag %d & local id %d\n",anchor_point_tag,atom->map(anchor_point_tag));

    int anchor_point_tmp = atom->map(anchor_point_tag);
    if (anchor_point_tmp < 0) {
      error->one(FLERR,"Fix entangle needs ghost atoms from further away");
    }
    int anchor_point = domain->closest_image(new_ent1,anchor_point_tmp);

    tagint dest1_tag = nvar[anchor_point][4];
    tagint dest2_tag = nvar[anchor_point][5];

    int dest1_tmp = atom->map(dest1_tag);
    int dest2_tmp = atom->map(dest2_tag);

    if (dest1_tmp < 0 || dest2_tmp < 0) {
      error->one(FLERR,"Fix entangle needs ghost atoms from further away");
    }
    int dest1 = domain->closest_image(new_ent1,dest1_tmp);
    int dest2 = domain->closest_image(new_ent1,dest2_tmp);

    if (nvar[new_ent1][2] == -1){
      if (nvar[anchor_point][2] == -1){
        nvar[new_ent1][0] = nvar[anchor_point][0]/2;
        nvar[new_ent1][1] = nvar[anchor_point][0]/2;
        // printf("FUCK MY LIFE 111\n\n");
      } else if (nvar[anchor_point][3] == -1){
        nvar[new_ent1][0] = nvar[anchor_point][1]/2;
        nvar[new_ent1][1] = nvar[anchor_point][1]/2;
        // printf("FUCK MY LIFE 111\n\n");
      }
    } else if (nvar[new_ent1][3] == -1){
      if (nvar[anchor_point][2] == -1){
        nvar[new_ent1][0] = nvar[anchor_point][0]/2;
        nvar[new_ent1][1] = nvar[anchor_point][0]/2;
        // printf("FUCK MY LIFE 111\n\n");
      } else if (nvar[anchor_point][3] == -1){
        nvar[new_ent1][0] = nvar[anchor_point][1]/2;
        nvar[new_ent1][1] = nvar[anchor_point][1]/2;
        // printf("FUCK MY LIFE 111\n\n");
      }
    }
    

    for (int jj = 0; jj < nlocal; jj++){
      if ((nvar[jj][2] == dest1_tag && nvar[jj][3] == dest2_tag) || (nvar[jj][3] == dest1_tag && nvar[jj][2] == dest2_tag)){
        new_ent2 = jj;
        break;
      }
    }

    if (update->ntimestep > 6143) {
    printf("\n\n    (PROC %d and TIMESTEP %ld) : Just created (%d & %d) Anchorpoint %d & destination (%d - %d) \n\n",me,update->ntimestep,tag[new_ent1],tag[new_ent2],tag[anchor_point],tag[dest1],tag[dest2]);
    }

    // calculate the number of monomers in destination sub-chain
//     printf("\n\nnvar of dest1 %d (tag %d)= [%f %f %f %f ...]\n\n",dest1,tag[dest1],nvar[dest1][0],nvar[dest1][1],nvar[dest1][2],nvar[dest1][3]);
//      printf("\n\nnvar of dest2 %d (tag %d)= [%f %f %f %f ...]\n\n",dest2,tag[dest2],nvar[dest2][0],nvar[dest2][1],nvar[dest2][2],nvar[dest2][3]);
    double shared_monomers = 0;
    if (nvar[dest1][2] == dest2_tag){
      shared_monomers = nvar[dest1][0];
    } else if (nvar[dest1][3] == dest2_tag){
      shared_monomers = nvar[dest1][1];
    }

    if (shared_monomers == 0) error->one(FLERR,"nvar data is not consistent (maybe)");

    nvar[new_ent2][0] = shared_monomers / 2;
    nvar[new_ent2][1] = shared_monomers / 2;

    process_created(new_ent1,new_ent2,2);

    if ((tag[new_ent1] == 714 && tag[new_ent2] == 2469) ||(tag[new_ent1] == 2469 && tag[new_ent2] == 714)){
      printf("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!         ERRORRR !!!!!!!!!!!!!!!!;\n\n");
    }

    nvar[new_ent1][4] = tag[new_ent2];
    nvar[new_ent2][4] = tag[new_ent1];

    //break;

  }

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds
  commflag = 3;
  comm->forward_comm(this);

  commflag = 1;
  comm->forward_comm(this,10);

  //printf("FUCK MY LIFE 222\n\n");

  update_topology();

  //if (update->ntimestep > 1630) printf("PROC %d HEREEEEEEEEEEE\n\n",me);
  MPI_Barrier(world);



  // update nall & nlocal
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;

  for (int k = 0; k < nall; k++){

    int new_atom1 = -1;
    int new_atom2 = -1;

    tagint new_atom1_tag = 0;
    tagint new_atom2_tag = 0;

    tagint seg_end1_tag = 0;
    tagint seg_end2_tag = 0;

    int seg_end1 = -1;
    int seg_end2 = -1;

    tagint anchor_atom_tag = 0;

    int anchor_atom = -1;

    int flag_pair_found = 0;

    if (molecule[k] == 0){
      if (nvar[k][2] == -1 || nvar[k][3] == -1){
        new_atom1 = k;
        // if (new_atom1 >= nlocal) continue;
        // printf("\n\n PROC %d - - NEW ATOM DAN ZAD DAN : %d with tag %d and num_bond %d\n\n",me,new_atom1,tag[new_atom1],num_bond[new_atom1]);
        //if (num_bond[new_atom1] != 1) error->one(FLERR,"Wrong bond on newly created particle");
        

        //for (int kk = 0; kk < num_bond[new_atom1]; kk++){
          int new_atom2_tmp = atom->map(nvar[new_atom1][4]);
          //if (bond_type[k][kk] == 2){
            new_atom2 = domain->closest_image(new_atom1,new_atom2_tmp);
          //}
        //}

        //printf("\n\nproc %d - new atoms = %d (tag %d) & %d\n\n",me,new_atom1,tag[new_atom1],bond_atom[new_atom1][0]);

        if (new_atom1 < 0 || new_atom2 < 0) error->one(FLERR,"new atoms not found");

        flag_pair_found = 1;

        new_atom1_tag = tag[new_atom1];
        new_atom2_tag = tag[new_atom2];

        seg_end1_tag = nvar[new_atom2][2];
        seg_end2_tag = nvar[new_atom2][3];

        int seg_end1_tmp = atom->map(seg_end1_tag);
        int seg_end2_tmp = atom->map(seg_end2_tag);

        if (seg_end1_tmp < 0 || seg_end2_tmp < 0) error->one(FLERR,"Some atom far away");

        seg_end1 = domain->closest_image(new_atom2,seg_end1_tmp);
        seg_end2 = domain->closest_image(new_atom2,seg_end2_tmp);

        if (nvar[new_atom1][2] == -1){
          anchor_atom_tag = nvar[new_atom1][3];
          int anchor_atom_tmp = atom->map(anchor_atom_tag);

          if (anchor_atom_tmp < 0) error->one(FLERR,"Some atom far away");

          anchor_atom = domain->closest_image(new_atom1,anchor_atom_tmp);
        } else if (nvar[new_atom1][3] == -1){
          anchor_atom_tag = nvar[new_atom1][2];
          int anchor_atom_tmp = atom->map(anchor_atom_tag);

          if (anchor_atom_tmp < 0) error->one(FLERR,"Some atom far away");

          anchor_atom = domain->closest_image(new_atom1,anchor_atom_tmp);
        }
      }
    }

    if (flag_pair_found != 1) continue;

    if (anchor_atom < nlocal){
      if (nvar[anchor_atom][2] == -1) {
        nvar[anchor_atom][2] = new_atom1_tag;
        nvar[anchor_atom][0] = nvar[new_atom1][0];
        nvar[anchor_atom][4] = 0;
        nvar[anchor_atom][5] = 0;
        nvar[anchor_atom][6] = 0;
        nvar[anchor_atom][7] = 0;
        nvar[anchor_atom][8] = 0;
      } else if (nvar[anchor_atom][3] == -1){
        nvar[anchor_atom][3] = new_atom1_tag;
        nvar[anchor_atom][1] = nvar[new_atom1][0];
        nvar[anchor_atom][4] = 0;
        nvar[anchor_atom][5] = 0;
        nvar[anchor_atom][6] = 0;
        nvar[anchor_atom][7] = 0;
        nvar[anchor_atom][8] = 0;
      }
    }

    if (seg_end1 < nlocal){
      if (nvar[seg_end1][2] == seg_end2_tag){
        nvar[seg_end1][2] = new_atom2_tag;
        nvar[seg_end1][0] = nvar[new_atom2][0];
      } else if(nvar[seg_end1][3] == seg_end2_tag){
        nvar[seg_end1][3] = new_atom2_tag;
        nvar[seg_end1][1] = nvar[new_atom2][0];
      }
    }

    if (seg_end2 < nlocal){
      if (nvar[seg_end2][2] == seg_end1_tag){
        nvar[seg_end2][2] = new_atom2_tag;
        nvar[seg_end2][0] = nvar[new_atom2][0];
      } else if(nvar[seg_end2][3] == seg_end1_tag){
        nvar[seg_end2][3] = new_atom2_tag;
        nvar[seg_end2][1] = nvar[new_atom2][0];
      }
    }
  }

  for (int n = 0; n < nlocal; n++){

    tagint connected_particle_tag = 0;
    int connected_particle = 0;

    if (molecule[n] == 0){
      nvar[n][4] = 0;
      if (nvar[n][2] == -1){
        connected_particle_tag = nvar[n][3];
        int connected_particle_tmp = atom->map(connected_particle_tag);
        if (connected_particle_tmp < 0) error->one(FLERR,"nakon namusan");

        connected_particle = domain->closest_image(n,connected_particle_tmp);

        molecule[n] = molecule[connected_particle];
      } else if (nvar[n][3] == -1){
        connected_particle_tag = nvar[n][2];
        int connected_particle_tmp = atom->map(connected_particle_tag);
        if (connected_particle_tmp < 0) error->one(FLERR,"nakon namusan");

        connected_particle = domain->closest_image(n,connected_particle_tmp);

        molecule[n] = molecule[connected_particle];
      } else if (nvar[n][3] != -1 && nvar[n][2] != -1){
        connected_particle_tag = nvar[n][2];
        int connected_particle_tmp = atom->map(connected_particle_tag);
        if (connected_particle_tmp < 0) error->one(FLERR,"nakon namusan");

        connected_particle = domain->closest_image(n,connected_particle_tmp);

        molecule[n] = molecule[connected_particle];
      }
    }
  }

  for (int ii = 0 ; ii < nlocal; ii++){
    nvar[ii][4] = 0;
    nvar[ii][5] = 0;
    nvar[ii][6] = 0;
    nvar[ii][7] = 0;
    nvar[ii][8] = 0;
  }
  

  commflag = 1;
  comm->forward_comm(this,10);

  MPI_Barrier(world);

  // update nbondlist

  nbondlist = neighbor->nbondlist;


  // looping to find bonds in destination subchains to break them

  for (int n = 0; n < nbondlist; n++){
    int i1 = bondlist[n][0];
    int i2 = bondlist[n][1];
    int type = bondlist[n][2];

    if (type == 2) continue;

    int i1_tag = tag[i1];
    int i2_tag = tag[i2];

    int i1_LHS = nvar[i1][2];
    int i1_RHS = nvar[i1][3];

    int i2_LHS = nvar[i2][2];
    int i2_RHS = nvar[i2][3];

    if ((i1_LHS == i2_tag || i1_RHS == i2_tag) && (i2_RHS == i1_tag || i2_LHS == i1_tag)){
      continue;
      
    } else {
      // printf("\n\ni1 (tag %d) = LHS :  %d     RHS :  %d\n\n",i1_tag,i1_LHS,i1_RHS);
      // printf("\n\ni2 (tag %d) = LHS :  %d     RHS :  %d\n\n",i2_tag,i2_LHS,i2_RHS);
      process_broken(i1,i2);
    }
  }

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds
  commflag = 3;
  comm->forward_comm(this);

  update_topology();


  // looping to find the atom which has a new bond in "nvar" but the bond is not created yet (each processor only does this for owned atoms)
  for (int k = 0; k < nlocal; k++){
    if (num_bond[k] > 2) continue;

    int found_left = 0;
    
    if (nvar[k][2] != -1){
      for (int mm = 0; mm < num_bond[k]; mm++){
        if (bond_atom[k][mm] == nvar[k][2]) found_left = 1;
      }
    } else if (nvar[k][2] == -1) found_left = 1;

    if (found_left == 0){
      int i1 = k;
      int i2 = atom->map(nvar[k][2]);
      if (i2 < 0) error->one(FLERR,"trying to access an atom far away");

      process_created(i1,i2,1);
      break;
    }

    int found_right = 0;

    if (nvar[k][3] != -1){
      for (int mm = 0; mm < num_bond[k]; mm++){
        if (bond_atom[k][mm] == nvar[k][3]) found_right = 1;
      }
    } else if (nvar[k][3] == -1) found_right = 1;

    if (found_right == 0){
      int i1 = k;
      int i2 = atom->map(nvar[k][3]);
      if (i2 < 0) error->one(FLERR,"trying to access an atom far away");

      process_created(i1,i2,1);
      break;
    }

  }

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds
  commflag = 3;
  comm->forward_comm(this);

  update_topology();


  MPI_Barrier(world);

  for (int i = 0; i < atom->nlocal; i++) {
    if(peratom_flag){
      array_atom[i][0] = nvar[i][0];
      array_atom[i][1] = nvar[i][1];
      array_atom[i][2] = nvar[i][2];
      array_atom[i][3] = nvar[i][3];
      array_atom[i][4] = nvar[i][4];
      array_atom[i][5] = nvar[i][5];
      array_atom[i][6] = nvar[i][6];
      array_atom[i][7] = nvar[i][7];
      array_atom[i][8] = nvar[i][8];
      array_atom[i][9] = nvar[i][9];
    }
  }


  next_reneighbor = update->ntimestep + 50;

  // DO NOT CONTINUE TO DISENTANGLEMENT (TEMPORARY FOR NOW)
  //return;






  // }


  /*--------------------------------------------------------------------------------------------------*/

  /*DISENTANGLEMENT SECTION*/

  // don't proceed if no disentanglement has been detected
  if (dis_flag == 0) return;

  // pair to be deleted
  tagint delete_ids[2] = {0 , 0};

  int ENT_pair;
  int ENT_pair_tmp;

  // looping to find a disentanglement
  for (int i = 0; i < nlocal; i++){
    if ((nvar[i][2] == -1 && nvar[i][0] < n_critical) || (nvar[i][3] == -1 && nvar[i][1] < n_critical)){
      //here a dangling end which is going to disentangle is identified "i"

      for (int j = 0; j < num_bond[i]; j++){
        if (bond_type[i][j]==2){
          ENT_pair_tmp = atom->map(bond_atom[i][j]);
          if (ENT_pair_tmp < 0) {
            error->one(FLERR,"Fix entangle needs ghost atoms from further away");
          }
          ENT_pair = domain->closest_image(i,ENT_pair_tmp);  
        }
      }

      delete_ids[0] = tag[i];
      delete_ids[1] = tag[ENT_pair];
      break;
    }
  }

  // only generate rand_number if delete_ids is assigned with non-zero entries
  double rand_number = 0;
  if (delete_ids[0] != 0 && delete_ids[1] != 0){
    rand_number = random->uniform();
  }

  struct {
    double random_num;
    int rank;
  } local, global;

  local.random_num = rand_number;
  local.rank = me;

  global.random_num = 0;
  global.rank = 0;

  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, world);

  if (global.random_num == 0){
    return;
    error->one(FLERR,"Inconsistent dis_flag has been set in previous timestep");
  } 

  int chosen_rank = global.rank;

  MPI_Bcast(&delete_ids,2,MPI_INT,chosen_rank,world);

  printf("\n\nAtoms to be deleted : %d & %d \n\n",delete_ids[0],delete_ids[1]);

  // at this point all processors have the same delete_ids

  update_nvar(delete_ids[0],delete_ids[1]);

  // for (int n = 0; n < nbondlist; n++){
  //   int i1 = bondlist[n][0];
  //   int i2 = bondlist[n][1];
  //   int type = bondlist[n][2];

  //   // delete any bond that has connection two any of delete_ids
  //   if (tag[i1] == delete_ids[0] || tag[i2] == delete_ids[0] || tag[i1] == delete_ids[1] || tag[i2] == delete_ids[1]){
  //     printf("\n\nBond broken between atoms %d & %d   - Proc %d\n\n",tag[i1],tag[i2],me);
  //     process_broken(i1,i2);
  //     bondlist[n][0] == -1;
  //     bondlist[n][1] == -1;
  //     bondlist[n][2] == -1;
  //   }

    
  // }

  for (int i = 0; i < nall; i++){
    for (int jj = 0; jj < num_bond[i]; jj++){
      if (bond_atom[i][jj] == delete_ids[0]){
        int i1 = i;
        int i2 = atom->map(delete_ids[0]);
        if (i2 < 0) error->one(FLERR,"Some atom is far away");
        printf("\n\nBond broken between atoms %d & %d   - Proc %d\n\n",tag[i1],tag[i2],me);
        process_broken(i1,i2);
      } else if (bond_atom[i][jj] == delete_ids[1]){
        int i1 = i;
        int i2 = atom->map(delete_ids[1]);
        if (i2 < 0) error->one(FLERR,"Some atom is far away");
        printf("\n\nBond broken between atoms %d & %d   - Proc %d\n\n",tag[i1],tag[i2],me);
        process_broken(i1,i2);
      }
    }
  }

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds
  commflag = 3;
  comm->forward_comm(this);

  update_topology();

  // only delete owned particles from delete_ids
  int local_id1 = atom->map(delete_ids[0]);
  int local_id2 = atom->map(delete_ids[1]);

  if (local_id1 < nlocal && local_id1 > -1) {
    AtomVec *avec = atom->avec;
    avec->copy(atom->nlocal - 1, local_id1, 1);
    atom->nlocal--;
    atom->natoms--;
  }

  if (local_id2 < nlocal && local_id2 > -1) {
    AtomVec *avec = atom->avec;
    avec->copy(atom->nlocal - 1, local_id2, 1);
    atom->nlocal--;
    atom->natoms--;
  }

  if (atom->map_style != Atom::MAP_NONE)  atom->map_init();
  // atom->natoms -= 2;
  

  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();


  // looping to find the atom which has a new bond in "nvar" but the bond is not created yet (each processor only does this for owned atoms)
  for (int k = 0; k < atom->nlocal; k++){
    if (num_bond[k] > 2) continue;

    int found_left = 0;
    
    if (nvar[k][2] != -1){
      for (int mm = 0; mm < num_bond[k]; mm++){
        if (bond_atom[k][mm] == nvar[k][2]) found_left = 1;
      }
    } else if (nvar[k][2] == -1) found_left = 1;

    if (found_left == 0){
      int i1 = k;
      int i2 = atom->map(nvar[k][2]);
      if (i2 < 0) error->one(FLERR,"trying to access an atom far away");

      process_created(i1,i2,1);
      printf("\n\nBond Created between atoms %d & %d   - Proc %d\n\n",tag[i1],tag[i2],me);
    }

    int found_right = 0;

    if (nvar[k][3] != -1){
      for (int mm = 0; mm < num_bond[k]; mm++){
        if (bond_atom[k][mm] == nvar[k][3]) found_right = 1;
      }
    } else if (nvar[k][3] == -1) found_right = 1;

    if (found_right == 0){
      int i1 = k;
      int i2 = atom->map(nvar[k][3]);
      if (i2 < 0) error->one(FLERR,"trying to access an atom far away");

      process_created(i1,i2,1);
      printf("\n\nBond Created between atoms %d & %d   - Proc %d\n\n",tag[i1],tag[i2],me);
    }

  }

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds

  commflag = 3;
  comm->forward_comm(this);

  update_topology();

    

  // trigger reneighboring
  next_reneighbor = update->ntimestep + 1;

}

/* ---------------------------------------------------------------------- */

 void FixEntangle::pre_force(int vflag)
{ 
  // return;
  // Main part of code
  // ATOM COUNTS
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // basic atom information
  double **x = atom->x;
  double **v = atom->v;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  double **f = atom->f;
  tagint *molecule = atom->molecule;


  // Possibly resize array "nu"
  if (atom->nmax > nmax) {
    memory->destroy(nu);
    nmax = atom->nmax;
    memory->create(nu,nmax,2,"entangle:nu");
  }

  // Possibly resize arrays "N_rest" & "N_0"
  memory->destroy(N_0);
  memory->destroy(N_rest);
  memory->create(N_0,nlocal,2,"entangle:N_0");
  memory->create(N_rest,nlocal,2,"entangle:N_rest");


  // Initialize array "nu"
  for (int i = 0; i < nall; i++){
    for (int j = 0; j < 2; j++){
      nu[i][j] = 0.0;
    }
  }

  for (int i = 0; i < nlocal; i++){
    for (int j = 0; j < 2; j++){
      N_0[i][j] = 0.0;
      N_rest[i][j] = 0.0;
    }
  }

  // Our state variable "nvar"
  double **nvar = atom->darray[index];

  // BOND INFORMATION
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  

  // DON'T PROCEED IF THE TIMESTEP IS NOT A MULTIPLE OF NEVERY
  if (update->ntimestep % nevery) return;
  v_init(vflag);

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm
  comm->forward_comm();


  // setting cross-linked entanglements

  if (update->ntimestep == 0){

    double Crosslinked_p = 0.5;

    int Crosslinked_n = floor(nlocal * Crosslinked_p);
    int Actualy_Crosslinked = 0;
    
    // for (int i = 0; i < nlocal; i++){
      // if (Actualy_Crosslinked >= Crosslinked_n) break;
      // if (nvar[i][9] == -1) continue;

      // int i1 = i;
      // int tag_i2 = 0;
      // for (int j = 0; j < num_bond[i1]; j++){
      //   if (bond_type[i1][j] == 2){
      //     tag_i2 = bond_atom[i1][j];
      //   }
      // }

      // if (tag_i2 == 0) error->one(FLERR,"Not found the ent pair");
    
      // int i2 = atom->map(tag_i2);

      // if (i2 < 0) error->one(FLERR,"Not found the ent pair");
      // if (i1 == i2) error->one(FLERR,"Wrong topology data");

      // if (i1 < nlocal && i2 < nlocal){
      //   nvar[i1][9] = -1;
      //   nvar[i2][9] = -1;
      //   Actualy_Crosslinked = Actualy_Crosslinked + 2;
      // }
  
    // }
    // for (int i = 0; i < nlocal; i++){
    //   if (nvar[i][2] == -1 || nvar[i][3] == -1) nvar[i][9] = -1;
    // }

    // printf("\n\nCrosslinked count 1: %d   (Proc %d & nlocal = %d) \n\n", Actualy_Crosslinked,me,nlocal);
    // reverse communication of nvar so ghost atoms get flagged if they are locked covalent crosslinks
    // commflag = 1;
    // comm->forward_comm(this,10);

    int ghofl = 0;
    for (int i = 0; i < nlocal; i++){
      if (nvar[i][9] == -1) ghofl++;
    }
    printf("\n\nCrosslinked count 2: %d   (Proc %d & nlocal = %d) \n\n", ghofl,me,nlocal);
  }
  

  // used later to average sliding rate
  double AVGnu=0;

  // used later for passing virial contribution of fix entangle
  double P[6];

  // aquire timestep size for integration
  double dt = update->dt;

  // initialize for later use of dumping nvar in a way to visualize tension of chains
  for (int i = 0; i < nlocal; i++){
    if (printcounter == 1){
        N_0[i][0] = nvar[i][0]/N_rest[i][0];
        N_0[i][1] = nvar[i][1]/N_rest[i][1];
    }
  }

  //loop over all entanglement points to calculate forces and then the monomer sliding rates
  for (int i = 0; i < nlocal; i++) {

    // Check if the atom is in the correct group... keep for generality
    if (!(mask[i] & groupbit)) continue;

    //We need a loop to go over columns of bond_atom because each atom is connected to three/one particles to find the left-hand side atom and right-hand side atom to aquire their vectorial distances.
    //We have used the tagID here to find the previous and next atoms (stored in nvar[*][3] for each atom)
    int LHS_atom = -1;
    int RHS_atom = -1;

    if (nvar[i][2] != -1){
      int LHS_atom_tmp = atom->map(nvar[i][2]);
      if (LHS_atom_tmp < 0) {
          error->one(FLERR,"Fix entangle needs ghost atoms from further away");
      }
      LHS_atom = domain->closest_image(i,LHS_atom_tmp);
    }

    if (nvar[i][3] != -1){
      int RHS_atom_tmp = atom->map(nvar[i][3]);
      if (RHS_atom_tmp < 0) {
          error->one(FLERR,"Fix entangle needs ghost atoms from further away");
      }
      RHS_atom = domain->closest_image(i,RHS_atom_tmp);
    }

    // checking if prev and next atoms are right
    if (nvar[i][2] != -1){
      if (LHS_atom == -1){
        error->one(FLERR,"Inconsistent finding of previous atom");
      }
    }

    if (nvar[i][3] != -1){
      if (RHS_atom == -1){
        error->one(FLERR,"Inconsistent finding of next atom");
      }
    }
    
    // components of left-hand side chain vector
    double delx1 = 0;
    double dely1 = 0;
    double delz1 = 0;

    double r1 = 0;
    
    if (nvar[i][2] != -1){
      delx1 = x[LHS_atom][0] - x[i][0];
      dely1 = x[LHS_atom][1] - x[i][1];
      delz1 = x[LHS_atom][2] - x[i][2];
      domain->minimum_image(delx1, dely1, delz1);
      double rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
      r1=sqrt(rsq1);
    } else {    // if a particle has dangling end on it's left
      r1 = b * pow(nvar[i][0],0.6);
      delx1 = sqrt(2)/2 * r1;
      dely1 = sqrt(2)/2 * r1;
      delz1 = 0;
    }

    // components of right-hand side chain vector
    double delx2 = 0;
    double dely2 = 0;
    double delz2 = 0;

    double r2 = 0;

    if (nvar[i][3] != -1){   
      delx2 = x[RHS_atom][0] - x[i][0];
      dely2 = x[RHS_atom][1] - x[i][1];
      delz2 = x[RHS_atom][2] - x[i][2];
      domain->minimum_image(delx2, dely2, delz2);
      double rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
      r2=sqrt(rsq2);
    } else {     // if a particle has dangling end on it's right
      r2 = b * pow(nvar[i][1],0.6);
      delx2 = sqrt(2)/2 * r2;
      dely2 = sqrt(2)/2 * r2;
      delz2 = 0;
    }

    //calculate the number of monomers for a sub-chain of length r1 to be at equilibrium
    N_rest[i][0] = pow(r1/b,1.667);
    N_rest[i][1] = pow(r2/b,1.667);

    // here we calculate the force components from the left and right hand side sub-chains 
    double fbond1x = 3 * delx1 / (nvar[i][0] * b * b) - 3 * (b * b * b) * nvar[i][0] * nvar[i][0] / (r1 * r1 * r1 * r1) * delx1/r1;
    double fbond1y = 3 * dely1 / (nvar[i][0] * b * b) - 3 * (b * b * b) * nvar[i][0] * nvar[i][0] / (r1 * r1 * r1 * r1) * dely1/r1;
    double fbond1z = 3 * delz1 / (nvar[i][0] * b * b) - 3 * (b * b * b) * nvar[i][0] * nvar[i][0] / (r1 * r1 * r1 * r1) * delz1/r1;

    double fbond2x = 3 * delx2 / (nvar[i][1] * b * b) - 3 * (b * b * b) * nvar[i][1] * nvar[i][1] / (r2 * r2 * r2 * r2) * delx2/r2;
    double fbond2y = 3 * dely2 / (nvar[i][1] * b * b) - 3 * (b * b * b) * nvar[i][1] * nvar[i][1] / (r2 * r2 * r2 * r2) * dely2/r2;
    double fbond2z = 3 * delz2 / (nvar[i][1] * b * b) - 3 * (b * b * b) * nvar[i][1] * nvar[i][1] / (r2 * r2 * r2 * r2) * delz2/r2;
    


    double fbond1 = sqrt(fbond1x * fbond1x + fbond1y * fbond1y + fbond1z * fbond1z);
    double fbond2 = sqrt(fbond2x * fbond2x + fbond2y * fbond2y + fbond2z * fbond2z);

    if ((fbond1x * delx1) < 0) fbond1 = 0;
    if ((fbond2x * delx2) < 0) fbond2 = 0;


    // here we pass the sum of the forces on each particle to the f[] array for use of other fixes
    // we also calculate the virial contributions and pass them via vtally()
    f[i][0] += (fbond1x + fbond2x);
    f[i][1] += (fbond1y + fbond2y);
    f[i][2] += (fbond1z + fbond2z);


    if (evflag) {
      P[0] -= ((delx1*fbond1x) + (delx2*fbond2x));
      P[1] -= ((dely1*fbond1y) + (dely2*fbond2y));
      P[2] -= ((delz1*fbond1z) + (delz2*fbond2z));
      P[3] -= ((delx1*fbond1y) + (delx2*fbond2y));
      P[4] -= ((delx1*fbond1z) + (delx2*fbond2z));
      P[5] -= ((dely1*fbond1z) + (dely2*fbond2z));

      v_tally(i, P);
    }

 

    // calculation of osmossic pressure
    // double Pi_1 = -1.5 * r1 * r1 / (b * b) * 1 / (nvar[i][0] * nvar[i][0]);/* + 2 * b * b * b * nvar[i][0] / (r1 * r1 * r1);*/
    // double Pi_2 = -1.5 * r2 * r2 / (b * b) * 1 / (nvar[i][1] * nvar[i][1]);/* + 2 * b * b * b * nvar[i][1] / (r2 * r2 * r2);*/

    double Pi_m = fbond2 - fbond1;
    // difference of tension from two sides is used to calculate monomer sliding rate at that entanglement
    double nu_rate = Pi_m / zeta; 

    // dis_flag is used to see if pre_exchange should be called at next timestep
    dis_flag = 0;

    // do not allow sliding at dangling ends if the number of monomers in reserve is less than a threshold "n_critical" (that entanglement will be disentangled at some point eventually)
    // this also turns on the dis_flag which allows disentanglement loop to be ran

    if (nvar[i][2] == -1){
      if (nvar[i][0] < n_critical){
        nu_rate = 0;
      }
    }

    if (nvar[i][3] == -1){
      if (nvar[i][1] < n_critical){
        nu_rate = 0;
      }
    }

    if (nvar[i][9] == -1) nu_rate = 0;

    if (nvar[i][0] < n_critical && nvar[i][2] == -1) nu_rate = 0;
    if (nvar[i][1] < n_critical && nvar[i][3] == -1) nu_rate = 0;

    nu[i][0] += (nu_rate * (-1));
    nu[i][1] += (nu_rate * (+1));  
    

    if (nvar[i][2] != -1){
      nu[LHS_atom][1] += (nu_rate * (-1));
    }

    if (nvar[i][3] != -1){
      nu[RHS_atom][0] += (nu_rate * (+1));
    }
    
    // Average sliding magnitude (sometimes printed as a measure of relaxation)
    AVGnu = AVGnu + (sqrt(nu_rate*nu_rate))/nlocal; 
  }

  if (update->ntimestep % 10000 == 0){
    printf("\nAverage sliding rate = %f\n",AVGnu);
  }

  // reverse communication of nu so ghost atoms aquire their sliding rates
  commflag = 1;
  comm->reverse_comm(this,2);
  
  // actual sliding part
  if (update->ntimestep > 200000){
    for (int j = 0; j < nlocal; j++) {
      nvar[j][0] = nvar[j][0] + nu[j][0] * update->dt;
      nvar[j][1] = nvar[j][1] + nu[j][1] * update->dt;
    }
  }


  // forward communication of nvar so other processors update their ghosts
  commflag = 1;
  comm->forward_comm(this,10);


  // array_atom is a per-atom array which can be dumped for visualization purposes in ovito
  for (int i = 0; i < nlocal; i++) {
    if(peratom_flag){
      array_atom[i][0] = nvar[i][0];
      array_atom[i][1] = nvar[i][1];
      array_atom[i][2] = nvar[i][2];
      array_atom[i][3] = nvar[i][3];
      array_atom[i][4] = nvar[i][4];
      array_atom[i][5] = nvar[i][5];
      array_atom[i][6] = nvar[i][6];
      array_atom[i][7] = nvar[i][7];
      array_atom[i][8] = nvar[i][8];
      array_atom[i][9] = nvar[i][9];
    }
  }

  printcounter = printcounter + 1;

  // Used to print custom stuff
  if ((printcounter % 10000)==0){
    // ......
  }

  // do not allow any disentanglement to happen at (timestep = 0)
  if (update->ntimestep == 0) return;


  // checking if any potential disentanglement case exisits in owned particles
  int dis_flag_local = 0;
  for (int j = 0; j < nlocal; j++){
    if (nvar[j][2] == -1 && nvar[j][0] < n_critical) dis_flag_local = 1;
    if (nvar[j][3] == -1 && nvar[j][1] < n_critical) dis_flag_local = 1;

    if (dis_flag_local) break;
  }


  // checking if other processors have seen a local dis flag

  MPI_Allreduce(&dis_flag_local, &dis_flag, 1, MPI_INT, MPI_MAX, world);
  
  // call pre_exchange at next timestep
  if (dis_flag == 1){
    next_reneighbor = update->ntimestep + 1;
  }
  
  
}

/* ---------------------------------------------------------------------- */

void FixEntangle::update_nvar(tagint id1, tagint id2){


  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;

  // per-atom state variable
  double **nvar = atom->darray[index];

  // BOND INFORMATION
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;

  // bond list from neighbor class
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  int local_id1 = atom->map(id1);
  int local_id2 = atom->map(id2);

  for (int i = 0; i < nlocal; i++){
    if (tag[i] == id1 || tag[i] == id2) continue;

    if (nvar[i][2] == id1){
      if (local_id1 >= 0){
        nvar[i][0] = nvar[i][0] + nvar[local_id1][0];
        nvar[i][2] = nvar[local_id1][2]; 
      } else {
        error->one(FLERR,"some atom far away");
      }
    } else if (nvar[i][2] == id2){
      if(local_id2 >= 0){
        nvar[i][0] = nvar[i][0] + nvar[local_id2][0];
        nvar[i][2] = nvar[local_id2][2]; 
      } else {
        error->one(FLERR,"some atom far away");
      }
    }

    if (nvar[i][3] == id1){
      if (local_id1 >= 0){
        nvar[i][1] = nvar[i][1] + nvar[local_id1][1];
        nvar[i][3] = nvar[local_id1][3]; 
      } else {
        error->one(FLERR,"some atom far away");
      }
    } else if (nvar[i][3] == id2){
      if(local_id2 >= 0){
        nvar[i][1] = nvar[i][1] + nvar[local_id2][1];
        nvar[i][3] = nvar[local_id2][3]; 
      } else {
        error->one(FLERR,"some atom far away");
      }
    }

  }


  commflag = 1;
  comm->forward_comm(this,10);


}

/* ---------------------------------------------------------------------- */

void FixEntangle::process_broken(int i, int j)
{

 // First add the pair to new_broken_pairs
  auto tag_pair = std::make_pair(atom->tag[i], atom->tag[j]);
  new_broken_pairs.push_back(tag_pair);

  // Manually search and remove from atom arrays
  // need to remove in case special bonds arrays rebuilt
  int nlocal = atom->nlocal;

  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *num_bond = atom->num_bond;

  if (i < nlocal) {
    int n = num_bond[i];

    int done = 0;
    for (int m = 0; m < n; m++) {
      if (bond_atom[i][m] == tag[j]) {
        for (int k = m; k < n - 1; k++) {
          bond_type[i][k] = bond_type[i][k + 1];
          bond_atom[i][k] = bond_atom[i][k + 1];
        }
        num_bond[i]--;
        break;
      }
      if (done) break;
    }
  }

  if (j < nlocal) {
    int n = num_bond[j];

    int done = 0;
    for (int m = 0; m < n; m++) {
      if (bond_atom[j][m] == tag[i]) {
        for (int k = m; k < n - 1; k++) {
          bond_type[j][k] = bond_type[j][k + 1];
          bond_atom[j][k] = bond_atom[j][k + 1];
        }
        num_bond[j]--;
        break;
      }
      if (done) break;
    }
  }

  // Update special neighbor list
  tagint *slist;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // remove i from special bond list for atom j and vice versa
  // ignore n2, n3 since 1-3, 1-4 special factors required to be 1.0
  if (i < nlocal) {
    slist = special[i];
    int n1 = nspecial[i][0];
    int m;
    for (m = 0; m < n1; m++)
      if (slist[m] == tag[j]) break;
    for (; m < n1 - 1; m++) slist[m] = slist[m + 1];
    nspecial[i][0]--;
    nspecial[i][1] = nspecial[i][2] = nspecial[i][0];
  }

  if (j < nlocal) {
    slist = special[j];
    int n1 = nspecial[j][0];
    int m;
    for (int m = 0; m < n1; m++)
      if (slist[m] == tag[i]) break;
    for (; m < n1 - 1; m++) slist[m] = slist[m + 1];
    nspecial[j][0]--;
    nspecial[j][1] = nspecial[j][2] = nspecial[j][0];
  }


}

/* --------------------------------------------------------------------- */

void FixEntangle::process_created(int i, int j, int n)
{

  // First add the pair to new_created_pairs
  auto tag_pair = std::make_pair(atom->tag[i], atom->tag[j]);
  new_created_pairs.push_back(tag_pair);

  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *num_bond = atom->num_bond;

  int nlocal = atom->nlocal;

  // Add bonds to atom class for i and j
  if (i < nlocal) {
    // if (num_bond[i] == atom->bond_per_atom)
    //   error->one(FLERR,"New bond exceeded bonds per atom in fix bond/dynamic");
    bond_type[i][num_bond[i]] = n;
    bond_atom[i][num_bond[i]] = atom->tag[j];
    num_bond[i]++;
  }

  if (j < nlocal) {
    // if (num_bond[j] == atom->bond_per_atom)
    //   error->one(FLERR,"New bond exceeded bonds per atom in fix bond/dynamic");
    bond_type[j][num_bond[j]] = n;
    bond_atom[j][num_bond[j]] = atom->tag[i];
    num_bond[j]++;
  }

  // add i to special bond list for atom j and vice versa
  // ignore n2, n3 since 1-3, 1-4 special factors required to be 1.0

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  if (i < nlocal) {
    int n1 = nspecial[i][0];
    if (n1 >= atom->maxspecial)
      error->one(FLERR, "Special list size exceeded in fix bond/dynamic");
    special[i][n1] = atom->tag[j];
    nspecial[i][0] += 1;
    nspecial[i][1] = nspecial[i][2] = nspecial[i][0];
  }

  if (j < nlocal) {
    int n1 = nspecial[j][0];
    if (n1 >= atom->maxspecial)
      error->one(FLERR, "Special list size exceeded in fix bond/dynamic");
    special[j][n1] = atom->tag[i];
    nspecial[j][0] += 1;
    nspecial[j][1] = nspecial[j][2] = nspecial[j][0];
  }

}

/* ----------------------------------------------------------------------
  Update special lists for recently broken/created bonds
  Assumes appropriate atom/bond arrays were updated, e.g. had called
      neighbor->add_temporary_bond(i1, i2, btype);
------------------------------------------------------------------------- */

void FixEntangle::update_topology()
{

  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;

  // In theory could communicate a list of broken bonds to neighboring processors here
  // to remove restriction that users use Newton bond off

  for (int ilist = 0; ilist < neighbor->nlist; ilist++) {
    NeighList *list = neighbor->lists[ilist];

    // Skip copied lists, will update original
    if (list->copy) continue;

    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    for (auto const &it : new_broken_pairs) {
      tagint tag1 = it.first;
      tagint tag2 = it.second;
      int i1 = atom->map(tag1);
      int i2 = atom->map(tag2);

      if (i1 < 0 || i2 < 0) {
        error->one(FLERR,"Fix bond/dynamic needs ghost atoms "
                    "from further away");
      }

      // Loop through atoms of owned atoms i j
      if (i1 < nlocal) {
        int *jlist = firstneigh[i1];
        int jnum = numneigh[i1];
        for (int jj = 0; jj < jnum; jj++) {
          int j = jlist[jj];
          j &= SPECIALMASK;    // Clear special bond bits
          if (tag[j] == tag2) jlist[jj] = j;
        }
      }

      if (i2 < nlocal) {
        int *jlist = firstneigh[i2];
        int jnum = numneigh[i2];
        for (int jj = 0; jj < jnum; jj++) {
          int j = jlist[jj];
          j &= SPECIALMASK;    // Clear special bond bits
          if (tag[j] == tag1) jlist[jj] = j;
        }
      }
    }
  }

  for (int ilist = 0; ilist < neighbor->nlist; ilist++) {
    NeighList *list = neighbor->lists[ilist];

    // Skip copied lists, will update original
    if (list->copy) continue;

    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    for (auto const &it : new_created_pairs) {
      tagint tag1 = it.first;
      tagint tag2 = it.second;
      int i1 = atom->map(tag1);
      int i2 = atom->map(tag2);

      if (i1 < 0 || i2 < 0) {
        error->one(FLERR,"Fix entangle needs ghost atoms "
                    "from further away");
      }

      // Loop through atoms of owned atoms i j
      if (i1 < nlocal) {
        int *jlist = firstneigh[i1];
        int jnum = numneigh[i1];
        for (int jj = 0; jj < jnum; jj++) {
          int j = jlist[jj];
          if (((j >> SBBITS) & 3) != 0) continue;               // Skip bonded pairs
          if (tag[j] == tag2) jlist[jj] = j ^ (1 << SBBITS);    // Add 1-2 special bond bits
        }
      }

      if (i2 < nlocal) {
        int *jlist = firstneigh[i2];
        int jnum = numneigh[i2];
        for (int jj = 0; jj < jnum; jj++) {
          int j = jlist[jj];
          if (((j >> SBBITS) & 3) != 0) continue;               // Skip bonded pairs
          if (tag[j] == tag1) jlist[jj] = j ^ (1 << SBBITS);    // Add 1-2 special bond bits
        }
      }
    }
  }

  new_broken_pairs.clear();
  new_created_pairs.clear();

}

/* ----------------------------------------------------------------------
   maxtag_all = current max atom ID for all atoms
------------------------------------------------------------------------- */

void FixEntangle::find_maxid()
{
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  tagint max = 0;
  for (int i = 0; i < nlocal; i++) max = MAX(max,tag[i]);
  MPI_Allreduce(&max,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixEntangle::grow_arrays(int nmax)
{
  if (peratom_flag) {
    memory->grow(array_atom,nmax,size_peratom_cols,"fix_entangle:array_atom");
  }
}
/* ---------------------------------------------------------------------- */

void FixEntangle::init_myarray()
{
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    for (int m = 0; m < size_peratom_cols; m++) {
      array_atom[i][m] = 0.0;
    }
  }
}

/* --------------------------------------------------------------------
 copy values within local atom-based arrays
----------------------------------------------------------------- */

void FixEntangle::copy_arrays(int i, int j, int delflag)
{
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[j][m] = array_atom[i][m];
  }
}
/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixEntangle::set_arrays(int i)
{
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[i][m] = 0;
  }
}
/* ---------------------------------------------------------------------- */

// THIS IS WHERE WE DEFINE COMMUNICATION //
int FixEntangle::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m,ns;

  m = 0;

  // CAN SET MULTIPLE COMMUNICATION TYPES USING commflag //
  if (commflag == 1) {

      double **nvar = atom->darray[index];

      for (i = 0; i < n; i++) {
        j = list[i];
        for (k = 0; k < 10; k++) {
          buf[m++] = nvar[j][k];
        }
      }
      return m;
  }

  if (commflag == 2) {
      int *num_bond = atom->num_bond;
      int **bond_type = atom->bond_type;
      tagint **bond_atom = atom->bond_atom;

      for (i = 0; i < n; i++) {
        j = list[i];
        ns = num_bond[j];
        buf[m++] = ubuf(ns).d;
        for (k = 0; k < ns; k++) {
          buf[m++] = ubuf(bond_type[j][k]).d;
          buf[m++] = ubuf(bond_atom[j][k]).d;
        }
      }
      return m;
  }

  if (commflag == 3){
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    for (int i = 0; i < n; i++) {
      int j = list[i];
      int ns = nspecial[j][0];
      buf[m++] = ubuf(ns).d;
      for (int k = 0; k < ns; k++) {
          buf[m++] = ubuf(special[j][k]).d;
      }
    }
    return m;
  }
}

/* ---------------------------------------------------------------------- */

// THIS IS THE OPPOSITE OF THE PREVIOUS SCRIPT //
// ALWAYS NEED TO DEFINE BOTH PACKING AND UNPACKING //
void FixEntangle::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    double **nvar = atom->darray[index];

    for (i = first; i < last; i++) {
        for (j = 0; j < 10; j++) {
          nvar[i][j] = buf[m++];
        }
    }
  } else if (commflag == 3) {

    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    for (int i = first; i < last; i++) {
      int ns = (int) ubuf(buf[m++]).i;
      nspecial[i][0] = ns;
      for (int j = 0; j < ns; j++) {
          special[i][j] = (tagint) ubuf(buf[m++]).i;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixEntangle::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  if (commflag == 1){
    for (i = first; i < last; i++) {
      for (int v = 0; v < 2; v++) {
          buf[m++] = nu[i][v];
      }
    }
    return m;
  }

  if (commflag == 2){
    double **nvar = atom->darray[index];
    for (i = first; i < last; i++) {
      for (int v = 0; v < 10; v++) {
          buf[m++] = nvar[i][v];
      }
    }
    return m;
  }
}

/* ---------------------------------------------------------------------- */

void FixEntangle::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  if (commflag == 1){
    for (i = 0; i < n; i++) {
      j = list[i];
      for (int v = 0; v < 2; v++) {
        nu[j][v] += buf[m++];
      }
    }
  }

  if (commflag == 2){
    double **nvar = atom->darray[index];
    for (i = 0; i < n; i++) {
      j = list[i];
      for (int v = 0; v < 10; v++) {
        nvar[j][v] = buf[m++];
      }
    }
  }
}
/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixEntangle::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(double);
  if (peratom_flag) bytes += (double)nmax*size_peratom_cols*sizeof(double);
  return bytes;
}
