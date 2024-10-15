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

#include "fix_slidering.h"

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

FixSlidering::FixSlidering(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), nu(nullptr), N_rest(nullptr), N_0(nullptr), random(nullptr)
{

  // IS THE NUMBER OF ARGUMENTS CORRECT? //
  if (narg < 2) error->all(FLERR,"Illegal fix slidering command");

  // WAS NEVERY SET CORRECTLY? //
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix slidering command");

  zeta = utils::numeric(FLERR,arg[4],false,lmp);
  if (zeta <= 0) error->all(FLERR,"Illegal fix slidering command - zeta should be greater than 0");

  n_critical = utils::numeric(FLERR,arg[5],false,lmp);
  if (n_critical <= 0) error->all(FLERR,"Illegal fix slidering command - critical segment count should be greater than 0");

  b = utils::numeric(FLERR,arg[6],false,lmp);
  if (b <= 0) error->all(FLERR,"Illegal fix slidering command - Kuhn length should be greater than 0");

  // MPI variables are introduced (Number of processors and Processor ID) //
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
  size_peratom_cols = 6;
  
  // used later to setup
  countflag = 0;

  // setting a seed for random number generation
  seed = 123455;

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

  // for allocating space to array_atom for fix slidering
  FixSlidering::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);

  if (peratom_flag) {
    init_myarray();
  }

  // Set forward communication size
  comm_forward = 1 + atom->maxspecial;
}

/* ---------------------------------------------------------------------- */

FixSlidering::~FixSlidering()
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

int FixSlidering::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSlidering::post_constructor()
{

  // CREATE A CALL TO FIX PROPERTY/ATOM //
  new_fix_id = utils::strdup(id + std::string("_FIX_PA"));
  modify->add_fix(fmt::format("{} {} property/atom d2_nvar_{} {} ghost yes",new_fix_id, group->names[igroup],id,std::to_string(6)));

  // RETURN THE INDEX OF OUR LOCALLY STORED ATOM ARRAY //
  int tmp1, tmp2;
  index = atom->find_custom(utils::strdup(std::string("nvar_")+id),tmp1,tmp2);

  // Create memory allocations
  nmax = atom->nmax;
  memory->create(nu,nmax,2,"slidring:nu");
  
}

/* ---------------------------------------------------------------------- */

void FixSlidering::init()
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
  // nvar = [ LHS n | RHS n | LHS tag | RHS tag | N/A | N/A ]

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

void FixSlidering::setup(int vflag)
{
  pre_force(vflag);
}

/* ----------------------------------------------------------------------
  perform disentanglement
  done before exchange, borders, reneighbor
  so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */
void FixSlidering::pre_exchange(){
  
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
            error->one(FLERR,"Fix slidering needs ghost atoms from further away");
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

 void FixSlidering::pre_force(int vflag)
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
    memory->create(nu,nmax,2,"slidering:nu");
  }

  // Possibly resize arrays "N_rest" & "N_0"
  memory->destroy(N_0);
  memory->destroy(N_rest);
  memory->create(N_0,nlocal,2,"slidering:N_0");
  memory->create(N_rest,nlocal,2,"slidering:N_rest");


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
          error->one(FLERR,"Fix slidering needs ghost atoms from further away");
      }
      LHS_atom = domain->closest_image(i,LHS_atom_tmp);
    }

    if (nvar[i][3] != -1){
      int RHS_atom_tmp = atom->map(nvar[i][3]);
      if (RHS_atom_tmp < 0) {
          error->one(FLERR,"Fix slidering needs ghost atoms from further away");
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

void FixSlidering::update_nvar(tagint id1, tagint id2){


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

void FixSlidering::process_broken(int i, int j)
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

void FixSlidering::process_created(int i, int j, int n)
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

void FixSlidering::update_topology()
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
        error->one(FLERR,"Fix slidering needs ghost atoms "
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

void FixSlidering::find_maxid()
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

void FixSlidering::grow_arrays(int nmax)
{
  if (peratom_flag) {
    memory->grow(array_atom,nmax,size_peratom_cols,"fix_slidering:array_atom");
  }
}
/* ---------------------------------------------------------------------- */

void FixSlidering::init_myarray()
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

void FixSlidering::copy_arrays(int i, int j, int delflag)
{
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[j][m] = array_atom[i][m];
  }
}
/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixSlidering::set_arrays(int i)
{
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[i][m] = 0;
  }
}
/* ---------------------------------------------------------------------- */

// THIS IS WHERE WE DEFINE COMMUNICATION //
int FixSlidering::pack_forward_comm(int n, int *list, double *buf,
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
void FixSlidering::unpack_forward_comm(int n, int first, double *buf)
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

int FixSlidering::pack_reverse_comm(int n, int first, double *buf)
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

void FixSlidering::unpack_reverse_comm(int n, int *list, double *buf)
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

double FixSlidering::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(double);
  if (peratom_flag) bytes += (double)nmax*size_peratom_cols*sizeof(double);
  return bytes;
}
