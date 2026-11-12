#ifndef MHD_RSOLVERS_HLLD_CRAW_HPP_
#define MHD_RSOLVERS_HLLD_CRAW_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlld_craw.hpp
//! \brief HLLD Riemann solver for MHD with cosmic rays (adiabatic part, paper Sec. 3.1).
//! Same structure as the fluid HLLD in hlld_mhd.hpp, with two additions:
//!   - total pressure includes the CR and wave pressure, P_cr + P_a+ + P_a-, in ptl/ptr;
//!   - the contact velocity u* (the middle wave speed) is written to uface_cr, and the
//!     adiabatic CR advective fluxes u* x (upwind value) to crflx_adv.
//! This is a separate function, called only when craw is on. hlld_mhd.hpp is not changed.
//! The signal speeds use the fluid fast speed as in the fluid HLLD; the effective-sound-speed
//! extension is applied separately.

#include "athena.hpp"
#include "eos/eos.hpp"
#include "craw/creos.hpp"

namespace mhd {

#define HLLD_CRAW_SMALL_NUMBER 1.0e-4

//----------------------------------------------------------------------------------------
//! \fn HLLD_CRAW<ivx>()
//! \brief HLLD for one face (m,k,j,i) with cosmic-ray states. Inputs are the fluid L/R
//! states wl, wr, bl, br, bx as in HLLD, and the CR L/R states wl_cr, wr_cr (craw slots).
//! Outputs: flx (fluid fluxes), ey/ez (EMFs), uface_cr (contact velocity), crflx_adv (CR
//! advective fluxes, nvars slots).
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLD_CRAW(const EOS_Data &eos, const CREOS_Data &creos,
               const int m, const int k, const int j, const int i,
               const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
               const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
               const DvceArray4D<Real> &bx,
               const DvceArray5D<Real> &wl_cr, const DvceArray5D<Real> &wr_cr,
               const DvceArray5D<Real> &flx,
               const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez,
               const DvceArray4D<Real> &uface_cr,
               const DvceArray5D<Real> &crflx_adv) {
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;
  Real spd[5];

  // ideal gas only (the paper's CR case); the fluid HLLD handles the isothermal case
  Real gm1 = eos.gamma - 1.0;
  Real igm1 = 1.0/gm1;

  //--- Step 1.  Load L/R states
  Real wl_idn = wl(m,IDN,k,j,i);
  Real wl_ivx = wl(m,ivx,k,j,i);
  Real wl_ivy = wl(m,ivy,k,j,i);
  Real wl_ivz = wl(m,ivz,k,j,i);
  Real wl_iby = bl(m,iby,k,j,i);
  Real wl_ibz = bl(m,ibz,k,j,i);

  Real wr_idn = wr(m,IDN,k,j,i);
  Real wr_ivx = wr(m,ivx,k,j,i);
  Real wr_ivy = wr(m,ivy,k,j,i);
  Real wr_ivz = wr(m,ivz,k,j,i);
  Real wr_iby = br(m,iby,k,j,i);
  Real wr_ibz = br(m,ibz,k,j,i);

  Real wl_ipr = eos.IdealGasPressure(wl(m,IEN,k,j,i));
  Real wr_ipr = eos.IdealGasPressure(wr(m,IEN,k,j,i));

  Real bxi = bx(m,k,j,i);

  Real bxsq = bxi*bxi;
  Real pbl = 0.5*(bxsq + (SQR(wl_iby) + SQR(wl_ibz)));
  Real pbr = 0.5*(bxsq + (SQR(wr_iby) + SQR(wr_ibz)));
  Real kel = 0.5*wl_idn*(SQR(wl_ivx) + (SQR(wl_ivy) + SQR(wl_ivz)));
  Real ker = 0.5*wr_idn*(SQR(wr_ivx) + (SQR(wr_ivy) + SQR(wr_ivz)));

  // CR pressure Pcr and total wave pressure Pa on each side, from the reconstructed states
  Real Pcr_l = creos.CRPressure(wl_cr(m,craw::ICRE,k,j,i));
  Real Pa_l  = creos.WavePressure(wl_cr(m,craw::IAWP,k,j,i))
             + creos.WavePressure(wl_cr(m,craw::IAWM,k,j,i));
  Real Pcr_r = creos.CRPressure(wr_cr(m,craw::ICRE,k,j,i));
  Real Pa_r  = creos.WavePressure(wr_cr(m,craw::IAWP,k,j,i))
             + creos.WavePressure(wr_cr(m,craw::IAWM,k,j,i));
  // total CR + wave pressure, used in ptl/ptr (paper Eqs. 230, 232)
  Real pcr_l = Pcr_l + Pa_l;
  Real pcr_r = Pcr_r + Pa_r;
  // effective pressure for the sound speed: gamma_th p_eff = gamma_th P_th + gamma_cr Pcr
  // + gamma_a Pa, the same form the craw timestep uses
  Real pl_eff = wl_ipr + (creos.gamma_cr*Pcr_l + creos.gamma_a*Pa_l)/eos.gamma;
  Real pr_eff = wr_ipr + (creos.gamma_cr*Pcr_r + creos.gamma_a*Pa_r)/eos.gamma;

  MHDCons1D ul, ur;
  ul.d  = wl_idn;
  ul.mx = wl_ivx*ul.d;
  ul.my = wl_ivy*ul.d;
  ul.mz = wl_ivz*ul.d;
  ul.e  = wl_ipr*igm1 + kel + pbl;
  ul.by = wl_iby;
  ul.bz = wl_ibz;

  ur.d  = wr_idn;
  ur.mx = wr_ivx*ur.d;
  ur.my = wr_ivy*ur.d;
  ur.mz = wr_ivz*ur.d;
  ur.e  = wr_ipr*igm1 + ker + pbr;
  ur.by = wr_iby;
  ur.bz = wr_ibz;

  //--- Step 2.  Signal speeds (fluid fast speed, as in the fluid HLLD)
  Real cfl = eos.IdealMHDFastSpeed(wl_idn, pl_eff, bxi, wl_iby, wl_ibz);
  Real cfr = eos.IdealMHDFastSpeed(wr_idn, pr_eff, bxi, wr_iby, wr_ibz);
  spd[0] = fmin(wl_ivx - cfl, wr_ivx - cfr);
  spd[4] = fmax(wl_ivx + cfl, wr_ivx + cfr);

  //--- Step 3.  L/R fluxes, with the CR and wave pressure in the total pressure
  Real ptl = wl_ipr + pbl + pcr_l;  // total pressures L,R, including CR and waves
  Real ptr = wr_ipr + pbr + pcr_r;

  MHDCons1D fl, fr, flxi;
  fl.d  = ul.mx;
  fl.mx = ul.mx*wl_ivx + ptl - bxsq;
  fl.my = ul.my*wl_ivx - bxi*ul.by;
  fl.mz = ul.mz*wl_ivx - bxi*ul.bz;
  fl.e  = wl_ivx*(ul.e + ptl - bxsq) - bxi*(wl_ivy*ul.by + wl_ivz*ul.bz);
  fl.by = ul.by*wl_ivx - bxi*wl_ivy;
  fl.bz = ul.bz*wl_ivx - bxi*wl_ivz;

  fr.d  = ur.mx;
  fr.mx = ur.mx*wr_ivx + ptr - bxsq;
  fr.my = ur.my*wr_ivx - bxi*ur.by;
  fr.mz = ur.mz*wr_ivx - bxi*ur.bz;
  fr.e  = wr_ivx*(ur.e + ptr - bxsq) - bxi*(wr_ivy*ur.by + wr_ivz*ur.bz);
  fr.by = ur.by*wr_ivx - bxi*wr_ivy;
  fr.bz = ur.bz*wr_ivx - bxi*wr_ivz;

  //--- Step 4.  Middle (contact) speed S_M and Alfven speeds
  Real sdl = spd[0] - wl_ivx;
  Real sdr = spd[4] - wr_ivx;
  spd[2] = (sdr*ur.mx - sdl*ul.mx + (ptl - ptr))/(sdr*ur.d - sdl*ul.d);

  Real sdml = spd[0] - spd[2];
  Real sdmr = spd[4] - spd[2];
  Real sdml_inv = 1.0/sdml;
  Real sdmr_inv = 1.0/sdmr;

  MHDCons1D ulst, uldst, urdst, urst;
  ulst.d = ul.d*sdl*sdml_inv;
  urst.d = ur.d*sdr*sdmr_inv;
  Real ulst_d_inv = 1.0/ulst.d;
  Real urst_d_inv = 1.0/urst.d;
  Real sqrtdl = sqrt(ulst.d);
  Real sqrtdr = sqrt(urst.d);

  spd[1] = spd[2] - fabs(bxi)/sqrtdl;
  spd[3] = spd[2] + fabs(bxi)/sqrtdr;

  //--- Step 5.  Intermediate states (same as the fluid HLLD)
  Real ptstl = ptl + ul.d*sdl*(spd[2] - wl_ivx);
  Real ptstr = ptr + ur.d*sdr*(spd[2] - wr_ivx);
  Real ptst = 0.5*(ptstr + ptstl);

  ulst.mx = ulst.d*spd[2];
  if (fabs(ul.d*sdl*sdml - bxsq) < HLLD_CRAW_SMALL_NUMBER*ptst) {
    ulst.my = ulst.d*wl_ivy;
    ulst.mz = ulst.d*wl_ivz;
    ulst.by = ul.by;
    ulst.bz = ul.bz;
  } else {
    Real tmp = bxi*(sdl - sdml)/(ul.d*sdl*sdml - bxsq);
    ulst.my = ulst.d*(wl_ivy - ul.by*tmp);
    ulst.mz = ulst.d*(wl_ivz - ul.bz*tmp);
    tmp = (ul.d*SQR(sdl) - bxsq)/(ul.d*sdl*sdml - bxsq);
    ulst.by = ul.by*tmp;
    ulst.bz = ul.bz*tmp;
  }
  Real vbstl = (ulst.mx*bxi + (ulst.my*ulst.by + ulst.mz*ulst.bz))*ulst_d_inv;
  ulst.e = (sdl*ul.e - ptl*wl_ivx + ptst*spd[2]
            + bxi*(wl_ivx*bxi + (wl_ivy*ul.by + wl_ivz*ul.bz) - vbstl))*sdml_inv;

  urst.mx = urst.d*spd[2];
  if (fabs(ur.d*sdr*sdmr - bxsq) < HLLD_CRAW_SMALL_NUMBER*ptst) {
    urst.my = urst.d*wr_ivy;
    urst.mz = urst.d*wr_ivz;
    urst.by = ur.by;
    urst.bz = ur.bz;
  } else {
    Real tmp = bxi*(sdr - sdmr)/(ur.d*sdr*sdmr - bxsq);
    urst.my = urst.d*(wr_ivy - ur.by*tmp);
    urst.mz = urst.d*(wr_ivz - ur.bz*tmp);
    tmp = (ur.d*SQR(sdr) - bxsq)/(ur.d*sdr*sdmr - bxsq);
    urst.by = ur.by*tmp;
    urst.bz = ur.bz*tmp;
  }
  Real vbstr = (urst.mx*bxi + (urst.my*urst.by + urst.mz*urst.bz))*urst_d_inv;
  urst.e = (sdr*ur.e - ptr*wr_ivx + ptst*spd[2]
            + bxi*(wr_ivx*bxi + (wr_ivy*ur.by + wr_ivz*ur.bz) - vbstr))*sdmr_inv;

  // double-star states (eqs. 59-63 of Miyoshi & Kusano), as in the fluid HLLD
  if (0.5*bxsq < HLLD_CRAW_SMALL_NUMBER*ptst) {
    uldst = ulst;
    urdst = urst;
  } else {
    Real invsumd = 1.0/(sqrtdl + sqrtdr);
    Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);
    uldst.d = ulst.d;
    urdst.d = urst.d;
    uldst.mx = ulst.mx;
    urdst.mx = urst.mx;

    Real tmp = invsumd*(sqrtdl*(ulst.my*ulst_d_inv) + sqrtdr*(urst.my*urst_d_inv)
                        + bxsig*(urst.by - ulst.by));
    uldst.my = uldst.d*tmp;
    urdst.my = urdst.d*tmp;

    tmp = invsumd*(sqrtdl*(ulst.mz*ulst_d_inv) + sqrtdr*(urst.mz*urst_d_inv)
                   + bxsig*(urst.bz - ulst.bz));
    uldst.mz = uldst.d*tmp;
    urdst.mz = urdst.d*tmp;

    tmp = invsumd*(sqrtdl*urst.by + sqrtdr*ulst.by
                   + bxsig*sqrtdl*sqrtdr*((urst.my*urst_d_inv) - (ulst.my*ulst_d_inv)));
    uldst.by = urdst.by = tmp;

    tmp = invsumd*(sqrtdl*urst.bz + sqrtdr*ulst.bz
                   + bxsig*sqrtdl*sqrtdr*((urst.mz*urst_d_inv) - (ulst.mz*ulst_d_inv)));
    uldst.bz = urdst.bz = tmp;

    tmp = spd[2]*bxi + (uldst.my*uldst.by + uldst.mz*uldst.bz)/uldst.d;
    uldst.e = ulst.e - sqrtdl*bxsig*(vbstl - tmp);
    urdst.e = urst.e + sqrtdr*bxsig*(vbstr - tmp);
  }

  //--- Step 6.  Scale the intermediate states into flux differences (as the fluid HLLD)
  uldst.d = spd[1]*(uldst.d - ulst.d);
  uldst.mx = spd[1]*(uldst.mx - ulst.mx);
  uldst.my = spd[1]*(uldst.my - ulst.my);
  uldst.mz = spd[1]*(uldst.mz - ulst.mz);
  uldst.e = spd[1]*(uldst.e - ulst.e);
  uldst.by = spd[1]*(uldst.by - ulst.by);
  uldst.bz = spd[1]*(uldst.bz - ulst.bz);

  ulst.d = spd[0]*(ulst.d - ul.d);
  ulst.mx = spd[0]*(ulst.mx - ul.mx);
  ulst.my = spd[0]*(ulst.my - ul.my);
  ulst.mz = spd[0]*(ulst.mz - ul.mz);
  ulst.e = spd[0]*(ulst.e - ul.e);
  ulst.by = spd[0]*(ulst.by - ul.by);
  ulst.bz = spd[0]*(ulst.bz - ul.bz);

  urdst.d = spd[3]*(urdst.d - urst.d);
  urdst.mx = spd[3]*(urdst.mx - urst.mx);
  urdst.my = spd[3]*(urdst.my - urst.my);
  urdst.mz = spd[3]*(urdst.mz - urst.mz);
  urdst.e = spd[3]*(urdst.e - urst.e);
  urdst.by = spd[3]*(urdst.by - urst.by);
  urdst.bz = spd[3]*(urdst.bz - urst.bz);

  urst.d = spd[4]*(urst.d - ur.d);
  urst.mx = spd[4]*(urst.mx - ur.mx);
  urst.my = spd[4]*(urst.my - ur.my);
  urst.mz = spd[4]*(urst.mz - ur.mz);
  urst.e = spd[4]*(urst.e - ur.e);
  urst.by = spd[4]*(urst.by - ur.by);
  urst.bz = spd[4]*(urst.bz - ur.bz);

  //--- Step 7.  Select the flux by the wave fan (as the fluid HLLD)
  if (spd[0] >= 0.0) {
    flxi = fl;
  } else if (spd[4] <= 0.0) {
    flxi = fr;
  } else if (spd[1] >= 0.0) {
    flxi.d = fl.d + ulst.d;   flxi.mx = fl.mx + ulst.mx;  flxi.my = fl.my + ulst.my;
    flxi.mz = fl.mz + ulst.mz; flxi.e = fl.e + ulst.e;    flxi.by = fl.by + ulst.by;
    flxi.bz = fl.bz + ulst.bz;
  } else if (spd[2] >= 0.0) {
    flxi.d = fl.d + ulst.d + uldst.d;   flxi.mx = fl.mx + ulst.mx + uldst.mx;
    flxi.my = fl.my + ulst.my + uldst.my; flxi.mz = fl.mz + ulst.mz + uldst.mz;
    flxi.e = fl.e + ulst.e + uldst.e;   flxi.by = fl.by + ulst.by + uldst.by;
    flxi.bz = fl.bz + ulst.bz + uldst.bz;
  } else if (spd[3] > 0.0) {
    flxi.d = fr.d + urst.d + urdst.d;   flxi.mx = fr.mx + urst.mx + urdst.mx;
    flxi.my = fr.my + urst.my + urdst.my; flxi.mz = fr.mz + urst.mz + urdst.mz;
    flxi.e = fr.e + urst.e + urdst.e;   flxi.by = fr.by + urst.by + urdst.by;
    flxi.bz = fr.bz + urst.bz + urdst.bz;
  } else {
    flxi.d = fr.d + urst.d;   flxi.mx = fr.mx + urst.mx;  flxi.my = fr.my + urst.my;
    flxi.mz = fr.mz + urst.mz; flxi.e = fr.e + urst.e;    flxi.by = fr.by + urst.by;
    flxi.bz = fr.bz + urst.bz;
  }

  // fluid fluxes and EMFs, same storage as the fluid HLLD
  flx(m,IDN,k,j,i) = flxi.d;
  flx(m,ivx,k,j,i) = flxi.mx;
  flx(m,ivy,k,j,i) = flxi.my;
  flx(m,ivz,k,j,i) = flxi.mz;
  flx(m,IEN,k,j,i) = flxi.e;
  ey(m,k,j,i) = -flxi.by;
  ez(m,k,j,i) =  flxi.bz;

  //--- Step 8.  CR face outputs (paper Sec. 3.1, adiabatic)
  // Contact velocity u*: the middle wave speed, the same value the star states use.
  uface_cr(m,k,j,i) = spd[2];

  // Advective CR fluxes u* x (upwind value), upwinded by the sign of u*. The reconstructed
  // left state of face i is wl(m,n,k,j,i) and the right state is wr(m,n,k,j,i), as the fluid
  // solver reads them. Slots are craw's: ICRE, ICRF, IAWP, IAWM.
  constexpr int ncr = craw::IAWM + 1;
  for (int n = 0; n < ncr; ++n) {
    Real upwind = (flxi.d >= 0.0) ? wl_cr(m,n,k,j,i) : wr_cr(m,n,k,j,i);
    crflx_adv(m,n,k,j,i) = spd[2]*upwind;
  }
}

} // namespace mhd
#endif // MHD_RSOLVERS_HLLD_CRAW_HPP_
