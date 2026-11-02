//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file craw_gaussian.cpp
//! \brief Problem generator for testing the craw (cosmic-ray + Alfven wave) module
//! on its own, with no hydro or MHD block in the input file.
//!
//! Initializes a 1-D Gaussian profile in the cosmic-ray energy density along x1 on top
//! of a uniform background. All other craw variables start at zero.
//!
//! Input parameters (<problem> block):
//!   amp     - peak amplitude of the Gaussian above the background (default 1.0)
//!   bkg     - uniform background value of the CR energy density (default 0.0)
//!   x0      - center of the Gaussian (default: domain midpoint)
//!   sigma   - standard deviation of the Gaussian, required (set in the input file)
//!
//! The Gaussian is written into the CR-energy index (craw::ICRE). The other craw
//! variables (flux, forward and backward wave energies) start at zero.

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "craw/craw.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::CRAWGaussian
//! \brief Problem generator for a 1-D Gaussian in the craw CR energy density

void ProblemGenerator::CRAWGaussian(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pcraw == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "craw_gaussian requires a <craw> block" << std::endl;
    exit(EXIT_FAILURE);
  }

  // read parameters
  Real amp = pin->GetOrAddReal("problem", "amp", 1.0);
  Real bkg = pin->GetOrAddReal("problem", "bkg", 0.0);
  auto &mesh_size = pmy_mesh_->mesh_size;
  Real x0 = pin->GetOrAddReal("problem", "x0",
                              0.5*(mesh_size.x1min + mesh_size.x1max));
  Real sigma = pin->GetReal("problem", "sigma");

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->pcraw->u0;
  // gamma_c for the initial flux, taken from the CR module so it matches the solver
  Real gamma_c = pmbp->pcraw->gamma_cr;

  // initialize: uniform background everywhere, Gaussian added to CR energy, rest zero
  par_for("pgen_craw_gauss", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je,
  is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x = CellCenterX(i-is, nx1, x1min, x1max);

    Real dx = x - x0;
    Real gauss = amp*exp(-0.5*dx*dx/(sigma*sigma));

    // E_c: Gaussian on the background. f_c: gamma_c * sign(x) * E_c.
    // E_+-: (1 + g(x)) * 1e-10, with g(x) the same Gaussian profile.
    Real Ec = bkg + gauss;
    Real sgn = (x >= x0) ? 1.0 : -1.0;
    u0(m,craw::ICRE,k,j,i) = Ec;
    u0(m,craw::ICRF,k,j,i) = gamma_c*sgn*Ec;
    u0(m,craw::IAWP,k,j,i) = (1.0 + gauss)*1.0e-6;
    u0(m,craw::IAWM,k,j,i) = (1.0 + gauss)*1.0e-6;
  });

  return;
}
