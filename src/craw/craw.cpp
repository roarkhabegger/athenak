//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file craw.cpp
//! \brief implementation of CosmicRayAW class constructor and destructor

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "craw/craw.hpp"

namespace craw {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

CosmicRayAW::CosmicRayAW(MeshBlockPack *ppack, ParameterInput *pin) :
    u0("craw_u0",1,1,1,1,1),
    u1("craw_u1",1,1,1,1,1),
    coarse_u0("craw_coarse_u0",1,1,1,1,1),
    uflx("craw_uflx",1,1,1,1,1),
    dtnew(0.0),
    pmy_pack(ppack) {
  // TODO(you): read <craw> parameters here
  feedback = pin->GetOrAddBoolean("craw","feedback",false);

  // allocate arrays for nvars variables over all meshblocks in this pack
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(u0, nmb, nvars, ncells3, ncells2, ncells1);
  Kokkos::realloc(u1, nmb, nvars, ncells3, ncells2, ncells1);
  Kokkos::realloc(uflx.x1f, nmb, nvars, ncells3, ncells2, ncells1+1);
  Kokkos::realloc(uflx.x2f, nmb, nvars, ncells3, ncells2+1, ncells1);
  Kokkos::realloc(uflx.x3f, nmb, nvars, ncells3+1, ncells2, ncells1);

  if (pmy_pack->pmesh->multilevel) {
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1);
  }

  // allocate boundary buffers for cell-centered variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_u->InitializeBuffers(nvars);
}

//----------------------------------------------------------------------------------------
// destructor

CosmicRayAW::~CosmicRayAW() {
  delete pbval_u;
}

} // namespace craw
