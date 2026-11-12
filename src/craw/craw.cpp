//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file craw.cpp
//! \brief implementation of CosmicRayAW class constructor and destructor

#include <cfloat>   // FLT_MIN
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "craw/craw.hpp"

namespace craw {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

CosmicRayAW::CosmicRayAW(MeshBlockPack *ppack, ParameterInput *pin) :
    u0("craw_u0",1,1,1,1,1),
    u1("craw_u1",1,1,1,1,1),
    coarse_u0("craw_coarse_u0",1,1,1,1,1),
    uflx("craw_uflx",1,1,1,1,1),
    uflx_aniso("craw_uflx_aniso",1,1,1,1,1),
    uface_cr("craw_uface_cr",1,1,1,1),
    crflx_adv("craw_crflx_adv",1,1,1,1,1),
    dtnew(0.0),
    subcycle_seconds(0.0),
    nsubcycles(0),
    pmy_pack(ppack) {
  
  Real l0 = 3.0856775809623245e18;
  Real t0 = 3.15576e13;  // s, 1 Myr in seconds
  Real true_c = 2.99792458e10  ;  // pc/Myr
  v_max = std::min(true_c/(l0/t0), pin->GetOrAddReal("craw","v_max",100.0));
  // lorentz and m_cr set the CR bin and do not change with the fluid
  lorentz = pin->GetOrAddReal("craw","lorentz",2.0);
  N_cr = pin->GetOrAddInteger("craw","ncr_sub",2);  // CR substeps per fluid step (even)
  // Strang split: N_cr/2 subcycles before and N_cr/2 after the fluid update, so N_cr must
  // be a positive even integer. Checked here so a bad value fails before the run starts.
  if (N_cr < 1 || N_cr % 2 != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<craw>/ncr_sub = " << N_cr << " must be a positive even integer"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  atol = pin->GetOrAddReal("craw","atol",1.0e-8);
  rtol = pin->GetOrAddReal("craw","rtol",1.0e-6);
  // CR pressures and floors live in the CR equation of state (creos.hpp)
  pcreos = new CREOS(pin);

  m_cr = pin->GetOrAddReal("craw","m_cr",1.0) * 1.67262192e-24;


  // v_A is needed only for standalone CR runs. With a fluid, the Alfven speed comes from
  // the fluid density and field, so v_A is not read.
  has_fluid = (pmy_pack->phydro != nullptr) || (pmy_pack->pmhd != nullptr);

  // With a fluid present, the CR coupling needs the magnetic field, so hydro-only is rejected.
  // A standalone CR run (no fluid) is still allowed.
  if (has_fluid && pmy_pack->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<craw> with a fluid requires <mhd>; hydro-only is not supported" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Unit scales and B-independent coefficients, in code units (L = pc, T = Myr, rho = m_p/cm^3).
  // The same constants serve the standalone branch (constant B, rho, T) and the MHD source step,
  // which evaluates chi and alpha per cell from the local state.
  if (pmy_pack->pmhd != nullptr && !pmy_pack->pmhd->peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<craw> with <mhd> requires an ideal-gas EOS (alpha uses P/rho)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  {
    Real rho0_g = 1.67262192e-24;      // g, density unit
    v_unit = l0/t0;                    // cm/s per code velocity
    E_unit = rho0_g*SQR(v_unit);       // erg/cm^3 per code energy density
    T_unit = t0;                       // s
    Real c_red_cgs = v_max*v_unit;     // reduced speed of light, cm/s
    // same factors as the standalone chi, with the B_gauss in the denominator left out
    chi_K = (3.0*M_PI/8.0)*4.803e-10/(lorentz*m_cr*c_red_cgs*c_red_cgs*c_red_cgs)
            * SQR(l0/t0)*t0*rho0_g*SQR(l0/t0);
    // same factors as the standalone alpha, with B_gauss and sqrt(kT/m_p) left out
    Real m_p_g = 1.67262192e-24;
    Real c_cgs = 2.99792458e10;
    alpha_K = (std::sqrt(M_PI)/4.0)*4.803e-10/(lorentz*m_p_g*c_cgs*c_cgs);
  }

  if (!has_fluid) {
    // code units: length = pc, time = Myr, density = m_p/cm^3 (so rho_unit = m_p in grams)
    // inputs: rho in code units; B in microgauss, converted to gauss
    Real rho0 = 1.67262192e-24;                       // g, density unit
    Real v_unit_cgs = l0/t0;  // cm/s, pc/Myr
    Real B_gauss = 1.0e-6*pin->GetReal("craw","B_muG");
    // Heaviside-Lorentz field (magnetic pressure B^2/2, as in the MHD solver):
    // B_HL = B_G/sqrt(4 pi), divided by the code pressure scale sqrt(rho_unit v_unit^2)
    B_code = B_gauss/std::sqrt(4.0*M_PI*rho0*SQR(l0/t0));
    // Alfven speed in code units: v_A = B/sqrt(rho)
    Real rho_code = pin->GetReal("craw","rho");
    v_A = B_code/std::sqrt(rho_code);
    Real c_red = v_max * l0/t0;
    chi_cr = (3.0*M_PI/8.0) * (4.803e-10)/(lorentz*m_cr*(c_red*c_red*c_red)* 1.0e-6*pin->GetReal("craw","B_muG"));
    chi_cr *= (SQR(l0/t0) *t0* rho0 * SQR(l0/t0));

    // Nonlinear Landau damping coefficient, self-consistent with the thermal gas (2010 Eq.
    // for Q_+-): alpha = (sqrt(pi)/4) (e / (gamma m c^2 B)) sqrt(k T / m_p), in cgs.
    // Temperature T (K) from <craw>, density drops out. Converted to code units with
    // E_unit * T_unit, E_unit = rho0 v_unit^2 with v_unit = l0/t0 (pc/Myr).
    Real T_gas = pin->GetOrAddReal("craw","T_gas",1.0e4);  // K
    Real kB = 1.380649e-16;                                 // erg/K
    Real e_esu = 4.803e-10;                                 // Gaussian charge
    Real c_cgs = 2.99792458e10;                             // cm/s
    Real m_p = 1.67262192e-24;                              // g
    Real alpha_cgs = (std::sqrt(M_PI)/4.0) * (e_esu/(lorentz*m_p*c_cgs*c_cgs*B_gauss))
                     * std::sqrt(kB*T_gas/m_p);             // cm s / g
    Real E_unit = rho0*SQR(l0/t0);                          // erg/cm^3
    Real T_unit = t0;                                       // s
    alpha_landau = alpha_cgs*E_unit*T_unit;                 // code units
  } else {
    // with a fluid, v_A, chi and alpha are set per cell in the source step from the MHD state
    v_A = 0.0;
    B_code = 0.0;
    chi_cr = 0.0;
    alpha_landau = 0.0;
  }

  // diagnostic: key CR parameters in code units, standalone runs only
  if (!has_fluid) {
    std::cout << "[craw] code units: L=pc, T=Myr, rho=m_p/cm^3" << std::endl;
    std::cout << "[craw] l0 (cm)        = " << l0 << std::endl;
    std::cout << "[craw] t0 (s)         = " << t0 << std::endl;
    std::cout << "[craw] velocity unit  = " << l0/t0 << " cm/s per code velocity" << std::endl;
    std::cout << "[craw] c (code vel)   = " << true_c/(l0/t0) << std::endl;
    std::cout << "[craw] v_max (code)   = " << v_max << std::endl;
    std::cout << "[craw] v_A (code)     = " << v_A << std::endl;
    std::cout << "[craw] rho (code)     = " << pin->GetReal("craw","rho") << std::endl;
    std::cout << "[craw] B_code         = " << B_code << std::endl;
    std::cout << "[craw] lorentz        = " << lorentz << std::endl;
    std::cout << "[craw] m_cr (cgs)     = " << m_cr << std::endl;
    std::cout << "[craw] chi (code)     = " << chi_cr << std::endl;
    std::cout << "[craw] alpha (1/Myr)  = " << alpha_landau << std::endl;
  }

  // chi = (9 pi/8) lorentz m c^3 B, with B the standalone field in code units

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
  Kokkos::realloc(uflx_aniso.x1f, nmb, nvars, ncells3, ncells2, ncells1+1);
  Kokkos::realloc(uflx_aniso.x2f, nmb, nvars, ncells3, ncells2+1, ncells1);
  Kokkos::realloc(uflx_aniso.x3f, nmb, nvars, ncells3+1, ncells2, ncells1);

  // adiabatic CR step: reconstructed L/R cell states and face outputs, all directions
  Kokkos::realloc(wl_cr, nmb, nvars, ncells3, ncells2, ncells1);
  Kokkos::realloc(wr_cr, nmb, nvars, ncells3, ncells2, ncells1);
  Kokkos::realloc(uface_cr.x1f, nmb, ncells3, ncells2, ncells1+1);
  Kokkos::realloc(uface_cr.x2f, nmb, ncells3, ncells2+1, ncells1);
  Kokkos::realloc(uface_cr.x3f, nmb, ncells3+1, ncells2, ncells1);
  Kokkos::realloc(crflx_adv.x1f, nmb, nvars, ncells3, ncells2, ncells1+1);
  Kokkos::realloc(crflx_adv.x2f, nmb, nvars, ncells3, ncells2+1, ncells1);
  Kokkos::realloc(crflx_adv.x3f, nmb, nvars, ncells3+1, ncells2, ncells1);
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
  delete pcreos;
}

} // namespace craw
