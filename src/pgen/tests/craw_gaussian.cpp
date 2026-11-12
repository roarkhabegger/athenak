//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file craw_gaussian.cpp
//! \brief 1-D test problem for the craw (cosmic-ray + Alfven wave) module, with an optional
//! CR impulse on a uniform background. Works with or without an <mhd> block.
//!
//! Uniform background (1-D, field along x1):
//!   rho  - gas density (default 1.0)
//!   vx   - gas velocity along x1 (default 0.0)
//!   Bx   - uniform magnetic field along x1 (default 0.0)
//!   pgas - gas thermal pressure (default 1.0)   [only written when <mhd> is present]
//!
//! CR background:
//!   bkg  - uniform background CR energy density E_c (default 0.0)
//!   bkgw - uniform background wave energy density E_+- (default 1.0e-6)
//!
//! CR impulse, selected by `impulse`, added to E_c on top of bkg:
//!   impulse = 0  none (uniform background only)
//!   impulse = 1  Gaussian, amplitude `amp`, standard deviation `sigma`, centred at `x0`
//!   impulse = 2  top hat, height `amp`, half-width `sigma`, centred at `x0`
//!   x0    - centre of the impulse (default: domain midpoint)
//!   amp   - peak height above bkg (default 1.0)
//!   sigma - Gaussian standard deviation, or top-hat half-width (required when impulse>0)
//!
//! Initial CR flux is the steady state of the flux equation (gyroresonant force balance,
//! time derivative zero), with chi and v_A from the source step's coefficients:
//!   f_cr = v_A gamma_cr E_c (E_a+ - E_a-)/(E_a+ + E_a-) - dP_cr/dx / (chi (E_a+ + E_a-))
//! with P_cr = (gamma_cr - 1) E_c and the gradient taken from the impulse profile.
//! The wave energies start at bkgw on both branches, so the streaming term is zero and the
//! flux is the diffusive part. Bx sets v_A and chi for both the MHD and standalone runs.

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "craw/craw.hpp"
#include "pgen/pgen.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::CRAWGaussian
//! \brief 1-D uniform background with an optional CR impulse (see file header)

//----------------------------------------------------------------------------------------
//! \fn CRAWImpulseProfile
//! \brief CR energy impulse profile at position x (zero for impulse = 0)

KOKKOS_INLINE_FUNCTION
Real CRAWImpulseProfile(int impulse, Real x, Real x0, Real amp, Real sigma) {
  Real dx = x - x0;
  if (impulse == 1) return amp*exp(-0.5*dx*dx/(sigma*sigma));
  if (impulse == 2) return (fabs(dx) <= sigma) ? amp : 0.0;
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn CRAWImpulseGrad
//! \brief Analytic dE/dx of the impulse profile at x. Gaussian: -amp (dx/sigma^2) G(x).
//! Top hat: zero in the interior (its edges are delta functions, not represented here).

KOKKOS_INLINE_FUNCTION
Real CRAWImpulseGrad(int impulse, Real x, Real x0, Real amp, Real sigma, Real dx) {
  Real d = x - x0;
  if (impulse == 1) return -amp*d/(sigma*sigma)*exp(-0.5*d*d/(sigma*sigma));
  if (impulse == 2) {
    return amp*((fabs(x0 - sigma - x) < 0.5*dx) - (fabs(x0 + sigma - x) < 0.5*dx))/dx;
  }
  return 0.0;
}

void ProblemGenerator::CRAWGaussian(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pcraw == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "craw_gaussian requires a <craw> block" << std::endl;
    exit(EXIT_FAILURE);
  }

  // uniform background
  Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  Real vx = pin->GetOrAddReal("problem", "vx", 0.0);
  Real bx = pin->GetOrAddReal("problem", "Bx", 0.0);
  Real pgas = pin->GetOrAddReal("problem", "pgas", 1.0);
  bool has_mhd = (pmbp->pmhd != nullptr);

  // CR background and impulse
  Real bkg = pin->GetOrAddReal("problem", "bkg", 0.0);
  Real bkgw = pin->GetOrAddReal("problem", "bkgw", 1.0e-6);
  int impulse = pin->GetOrAddInteger("problem", "impulse", 1);
  Real amp = pin->GetOrAddReal("problem", "amp", 1.0);
  auto &mesh_size = pmy_mesh_->mesh_size;
  Real x0 = pin->GetOrAddReal("problem", "x0",
                              0.5*(mesh_size.x1min + mesh_size.x1max));
  Real sigma = 0.0;
  if (impulse != 0) {
    sigma = pin->GetReal("problem", "sigma");
  }
  if (impulse < 0 || impulse > 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "problem/impulse must be 0 (none), 1 (Gaussian), 2 (top hat)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->pcraw->u0;

  // Steady-state CR flux (see header). The source step's coefficients, in code units:
  //   chi = chi_K / B_gauss, B_gauss = B_HL * sqrt(4 pi E_unit), so 1/chi = B_HL*Bfac/chi_K
  //   v_A = B_HL / sqrt(rho)
  Real gamma_c = pmbp->pcraw->gamma_cr;
  Real gcm1 = gamma_c - 1.0;
  Real Bfac = std::sqrt(4.0*M_PI*pmbp->pcraw->E_unit);
  Real inv_chi = bx*Bfac/pmbp->pcraw->chi_K;
  Real vA = bx/std::sqrt(rho);

  // CR: impulse on E_c, uniform background elsewhere. Written on craw's u0 directly.
  par_for("pgen_craw_cr", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je,
  is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x = CellCenterX(i-is, nx1, x1min, x1max);

    // steady-state flux: analytic gradient of the impulse profile
    Real Ec = bkg + CRAWImpulseProfile(impulse, x, x0, amp, sigma);
    Real dPcr = gcm1*CRAWImpulseGrad(impulse, x, x0, amp, sigma, size.d_view(m).dx1);
    Real Eap = bkgw, Eam = bkgw;
    Real wsum = Eap + Eam;
    Real f_stream = (wsum > 0.0) ? vA*gamma_c*Ec*(Eap - Eam)/wsum : 0.0;
    Real f_diff = (wsum > 0.0) ? -dPcr*inv_chi/wsum : 0.0;

    u0(m,craw::ICRE,k,j,i) = Ec;
    u0(m,craw::ICRF,k,j,i) = f_stream + f_diff;
    u0(m,craw::IAWP,k,j,i) = Eap;
    u0(m,craw::IAWM,k,j,i) = Eam;
  });

  // Gas: uniform primitives on w0 and uniform Bx on the faces, then the EOS builds the
  // conserved state, as the other pgens do (see linear_wave.cpp, diffusion.cpp).
  if (has_mhd) {
    auto &mhd = *pmbp->pmhd;
    auto &w0 = mhd.w0;
    auto &b0 = mhd.b0;
    auto &bcc0 = mhd.bcc0;
    auto &u0_mhd = mhd.u0;
    Real gam = mhd.peos->eos_data.gamma;
    Real gm1 = gam - 1.0;

    par_for("pgen_craw_gas", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je,
    is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      w0(m,IDN,k,j,i) = rho;
      w0(m,IVX,k,j,i) = vx;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IEN,k,j,i) = pgas/gm1;  // primitive IEN is internal energy
      bcc0(m,IBX,k,j,i) = bx;
      bcc0(m,IBY,k,j,i) = 0.0;
      bcc0(m,IBZ,k,j,i) = 0.0;
      b0.x1f(m,k,j,i) = bx;
      if (i == ie) { b0.x1f(m,k,j,i+1) = bx; }
    });
    mhd.peos->PrimToCons(w0, bcc0, u0_mhd, is, ie, js, je, ks, ke);
  }

  return;
}
