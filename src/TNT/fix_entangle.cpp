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
  Fix(lmp, narg, arg), nu(nullptr), N_rest(nullptr), N_0(nullptr)
{

  // IS THE NUMBER OF ARGUMENTS CORRECT? //
  if (narg < 2) error->all(FLERR,"Illegal fix entangle command");

  // WAS NEVERY SET CORRECTLY? //
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix entangle command");

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
  size_peratom_cols = 6;
  
  // used later to setup
  countflag = 0;

  // INITIALIZE ANY LOCAL ARRAYS //
  nu = nullptr;
  N_rest = nullptr;
  N_0 = nullptr;

  nmax = atom->nmax;

  // for allocating space to array_atom
  FixEntangle::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);

  if (peratom_flag) {
    init_myarray();
  }
}

/* ---------------------------------------------------------------------- */

FixEntangle::~FixEntangle()
{
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
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEntangle::post_constructor()
{
  // CREATE A CALL TO FIX PROPERTY/ATOM //
  new_fix_id = utils::strdup(id + std::string("_FIX_PA"));
  modify->add_fix(fmt::format("{} {} property/atom d2_nvar_{} {} ghost yes",new_fix_id, group->names[igroup],id,std::to_string(4)));

  // RETURN THE INDEX OF OUR LOCALLY STORED ATOM ARRAY //
  int tmp1, tmp2;
  index = atom->find_custom(utils::strdup(std::string("nvar_")+id),tmp1,tmp2);

  // nvar IS THE POINTER TO OUR STATE VARIABLE ARRAY! //
  double **nvar = atom->darray[index];
  
  // "printcounter" is later used for printing custom messages every Nth timestep
  int printcounter = 1;

  // accessing atom count data
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  
  int *mask = atom->mask;

  // accessing per-atom velocities and radius because the contain nvar (from data file)
  double **v = atom->v;
  double *radius = atom->radius;


  // ! INITIALIZE OUR NEW STATE VARIABLE ! //
  //   !! EACH ATOM WILL STORE EXACTLY 4 VALUES !!   //
  for (int i = 0; i < nall; i++) {
    for (int m = 0; m < 4; m++) {
      if (mask[i] & groupbit) { 
        nvar[i][m] = 0;
      }
    }
  }

  for (int i = 0; i < nlocal; i++) {
    nvar[i][0] = v[i][0];
    nvar[i][1] = v[i][1];
    nvar[i][2] = v[i][2];
    nvar[i][3] = radius[i]*2;

    v[i][0] = 0;
    v[i][1] = 0;
    v[i][2] = 0;
  }

  commflag = 1;
  comm->forward_comm(this,4);


  // Create memory allocations
  nmax = atom->nmax;
  memory->create(nu,nmax,2,"entangle:nu");
  memory->create(N_rest,nlocal,2,"entangle:N_rest");
  memory->create(N_0,nlocal,2,"entangle:N_0");
  
}

/* ---------------------------------------------------------------------- */

void FixEntangle::init()
{

}

/* ---------------------------------------------------------------------- */

void FixEntangle::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^verlet")){
    pre_force(vflag);
  }

  // Only run this in the beginning of simulation (transferring nvar data from velocities to actual peratom array)
  if (countflag) return;
  countflag = 1;

  // LOCATE THE POINTER TO OUR VARIABLE
  int tmp1, tmp2;
  double **nvar = atom->darray[index];

  // local atom count
  int nlocal = atom->nlocal;
  
  // [nvar] = [Leftside monomer count | Rightside monomer count | dangling end flag or Anchor point timer | chain tagID]
  // for (int i = 0; i < nlocal; i++) {
  //   nvar[i][0] = v[i][0];
  //   nvar[i][1] = v[i][1];
  //   nvar[i][2] = v[i][2];
  //   nvar[i][3] = radius[i]*2;
  // }

  // commflag = 1;
  // comm->forward_comm(this,4);

}

/* ---------------------------------------------------------------------- */

 void FixEntangle::pre_force(int vflag)
{
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

    //identifying if it is a dangling end 

    // Check if the atom is in the correct group... keep for generality
    if (!(mask[i] & groupbit)) continue;

    //We need a loop to go over columns of bond_atom because each atom is connected to three/one particles to find the left-hand side atom and right-hand side atom to aquire their vectorial distances.
    //We have used the tagID here to find the previous and next atoms (stored in nvar[*][3] for each atom)

    int LHS_atom = -1;
    int RHS_atom = -1;

    for (int jj=0; jj < num_bond[i]; jj++){
      
      int bonded_atom_tmp = atom->map(bond_atom[i][jj]);

      if (bonded_atom_tmp < 0) {
        error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
      }

      int bonded_atom = domain->closest_image(i,bonded_atom_tmp);
      
      if (bond_type[i][jj]==1 && nvar[bonded_atom][3]==nvar[i][3]-1){
        LHS_atom = bonded_atom;
      }

      if (bond_type[i][jj]==1 && nvar[bonded_atom][3]==nvar[i][3]+1){
        RHS_atom = bonded_atom;
      }
    }

    // checking if prev and next atoms are right

    if (nvar[i][2] != -1){
      if (LHS_atom == -1){
        error->one(FLERR,"Inconsistent finding of previous atom");
      }
    }

    if (nvar[i][2] != 1){
      if (RHS_atom == -1){
        error->one(FLERR,"Inconsistent finding of next atom");
      }
    }
    
    // components of left-hand side chain vector
    double delx1 = 0;
    double dely1 = 0;
    double delz1 = 0;

    // components of right-hand side chain vector
    double delx2 = 0;
    double dely2 = 0;
    double delz2 = 0;

    double r1 = 0;
    double r2 = 0;

    // defining kuhn length
    double b = 0.05;

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

    if (nvar[i][2] != 1){   
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


    // components of left-hand side force
    double fbond1x = 0;
    double fbond1y = 0;
    double fbond1z = 0;

    // components of right-hand side force
    double fbond2x = 0;
    double fbond2y = 0;
    double fbond2z = 0;

    // here we calculate the force components from the left and right hand side sub-chains 

    fbond1x = 3 * delx1 / (nvar[i][0] * b * b) - 3 * (b * b * b) * nvar[i][0] * nvar[i][0] / (r1 * r1 * r1 * r1) * delx1/r1;
    fbond1y = 3 * dely1 / (nvar[i][0] * b * b) - 3 * (b * b * b) * nvar[i][0] * nvar[i][0] / (r1 * r1 * r1 * r1) * dely1/r1;
    fbond1z = 3 * delz1 / (nvar[i][0] * b * b) - 3 * (b * b * b) * nvar[i][0] * nvar[i][0] / (r1 * r1 * r1 * r1) * delz1/r1;

    fbond2x = 3 * delx2 / (nvar[i][1] * b * b) - 3 * (b * b * b) * nvar[i][1] * nvar[i][1] / (r2 * r2 * r2 * r2) * delx2/r2;
    fbond2y = 3 * dely2 / (nvar[i][1] * b * b) - 3 * (b * b * b) * nvar[i][1] * nvar[i][1] / (r2 * r2 * r2 * r2) * dely2/r2;
    fbond2z = 3 * delz2 / (nvar[i][1] * b * b) - 3 * (b * b * b) * nvar[i][1] * nvar[i][1] / (r2 * r2 * r2 * r2) * delz2/r2;

    //Used to print custom stuff
    if ((printcounter % 10000)==0){
      // ......
    }
    
    // sliding friction coefficient
    double zeta = 1;

    // here we pass the sum of the forces on each particle to the f[] array for use of other fixes
    // we also calculate the virial contributions and pass them via vtally()
    f[i][0] += (fbond1x + fbond2x);
    f[i][1] += (fbond1y + fbond2y);
    f[i][2] += (fbond1z + fbond2z);


    if (evflag) {
      P[0] += ((delx1*fbond1x) + (delx2*fbond2x));
      P[1] += ((dely1*fbond1y) + (dely2*fbond2y));
      P[2] += ((delz1*fbond1z) + (delz2*fbond2z));
      P[3] += ((delx1*fbond1y) + (delx2*fbond2y));
      P[4] += ((delx1*fbond1z) + (delx2*fbond2z));
      P[5] += ((dely1*fbond1z) + (dely2*fbond2z));

      v_tally(i, P);
    }

    // force magnitudes
    double fbond1 = 0;
    double fbond2 = 0;

    // calculate the force magnitude from each side of the entanglement for calculation of sliding rate
    double fbond1_squared = (fbond1x * fbond1x) + (fbond1y * fbond1y) + (fbond1z * fbond1z);
    fbond1 = sqrt(fbond1_squared);
    
    // since the fbond1 might be a repulsive, we should check (delx1 is always calculated as it is a tensile vector so if fbondx1 and delx1 are in reverse direction it means fbond1 is repulsive)
    if (delx1*fbond1x<0){
      fbond1 = -fbond1;
    }

    double fbond2_squared = (fbond2x * fbond2x) + (fbond2y * fbond2y) + (fbond2z * fbond2z);
    fbond2 = sqrt(fbond2_squared);
    if (delx2*fbond2x<0){
      fbond2 = -fbond2;
    }

    // calculation of osmossic pressure
    double Pi_1, Pi_2;

    //Pi_1 = (-0.5 * (r1 / (nvar[i][0]*b)) * (r1 / (nvar[i][0]*b)) - log(1 - (r1 / (nvar[i][0]*b)) * (r1 / (nvar[i][0]*b))) + 2 * r1 * r1 / (r1 * r1 - nvar[i][0] * nvar[i][0] * b * b) + nvar[i][0] * b * b * b / (r1 * r1 * r1));
    Pi_1 = -3 * r1 * r1 / (2 * b * b) * 1 / (nvar[i][0] * nvar[i][0]) + 2 * b * b * b * nvar[i][0] / (r1 * r1 * r1);

    //Pi_2 = (-0.5 * (r2 / (nvar[i][1]*b)) * (r2 / (nvar[i][1]*b)) - log(1 - (r2 / (nvar[i][1]*b)) * (r2 / (nvar[i][1]*b))) + 2 * r2 * r2 / (r2 * r2 - nvar[i][1] * nvar[i][1] * b * b) + nvar[i][1] * b * b * b / (r2 * r2 * r2));
    Pi_2 = -3 * r2 * r2 / (2 * b * b) * 1 / (nvar[i][1] * nvar[i][1]) + 2 * b * b * b * nvar[i][1] / (r2 * r2 * r2) ;
    
    
    double Pi_m = Pi_2 - Pi_1;

    // difference of tension from two sides is used to calculate monomer sliding rate at that entanglement
    double nu_rate = Pi_m / zeta; 
    
    nu[i][0] += (nu_rate * (-1));
    nu[i][1] += (nu_rate * (+1));  
    

    if (nvar[i][2] != -1){
      nu[LHS_atom][1] += (nu_rate * (-1));
    }

    if (nvar[i][2] != 1){
      nu[RHS_atom][0] += (nu_rate * (+1));
    }
    
    // Average sliding magnitude (sometimes printed as a measure of relaxation)
    AVGnu = AVGnu + (sqrt(nu_rate*nu_rate))/nlocal; 
  }
  
  if (printcounter % 10000 == 0  || printcounter == 2){
  printf("average sliding rate : %f\n\n",AVGnu);
  }
  // reverse communication of nu so ghost atoms aquire their sliding rates
  comm->reverse_comm(this,2);

  for (int j = 0; j < nlocal; j++) {
    nvar[j][0] = nvar[j][0] + nu[j][0] * update->dt;
    nvar[j][1] = nvar[j][1] + nu[j][1] * update->dt;
  }


  //printf("\n\nAVERAGE SLIDING RATE  : %f \n\n",AVGnu);
  commflag = 1;
  comm->forward_comm(this,4);

  // array_atom is a per-atom array which can be dumped for visualization purposes in ovito
  for (int i = 0; i < nlocal; i++) {
    if(peratom_flag){
      if(N_rest[i][0] != 0){  
      // array_atom[i][0] = (nvar[i][0]/N_rest[i][0] - N_0[i][0]) * 1/(1-N_0[i][0]);
      // array_atom[i][0] = (nvar[i][0]/N_rest[i][0]);
      }
      if(N_rest[i][1] != 0){
      // array_atom[i][1] = (nvar[i][1]/N_rest[i][1] - N_0[i][1]) * 1/(1-N_0[i][1]);
      // array_atom[i][1] = (nvar[i][1]/N_rest[i][1]);
      }  
      array_atom[i][0] = nvar[i][0];
      array_atom[i][1] = nvar[i][1];
      array_atom[i][2] = nvar[i][2];
      array_atom[i][3] = nvar[i][3];
      array_atom[i][4] = (nu[i][0]);
      array_atom[i][5] = (nu[i][1]);
    }
  }
  
  printcounter = printcounter + 1;

  
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
        for (k = 0; k < 4; k++) {
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
        for (j = 0; j < 4; j++) {
          nvar[i][j] = buf[m++];
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

  for (i = first; i < last; i++) {
    for (int v = 0; v < 2; v++) {
        buf[m++] = nu[i][v];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixEntangle::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    for (int v = 0; v < 2; v++) {
      nu[j][v] += buf[m++];
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
