//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file craw_tasks.cpp
//! \brief task assembly and stub task functions for CosmicRayAW.
//! Every stub returns TaskStatus::complete and does nothing yet. Fill them in.

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "driver/driver.hpp"
#include "bvals/bvals.hpp"
#include "craw/craw.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "ion-neutral/ion-neutral.hpp"

namespace craw {
//----------------------------------------------------------------------------------------
//! \fn TaskID CosmicRayAW::CRAWFluidFluxDependency
//! \brief Returns the task that CalculateFluxes must wait for, so the CR flux is taken
//! from the same stage-start state as the fluid flux and before the fluid CT update.
//!   - two-fluid ion-neutral: the MHD flux runs after impl_2x, which sets b1 = b0 and
//!     does the implicit drag, so depend on impl_2x.
//!   - MHD only: depend on the MHD copyu, the predecessor of MHD's Fluxes.
//!   - hydro only: depend on the hydro copyu, the predecessor of Hydro's Fluxes.
//!   - none of these: fall back to this module's own copyu.

TaskID CosmicRayAW::CRAWFluidFluxDependency(TaskID &own_copyu) {
  if (pmy_pack->pionn != nullptr) {
    return pmy_pack->pionn->id.impl_2x;
  }
  if (pmy_pack->pmhd != nullptr) {
    return pmy_pack->pmhd->id.copyu;
  }
  if (pmy_pack->phydro != nullptr) {
    return pmy_pack->phydro->id.copyu;
  }
  return own_copyu;
}

//----------------------------------------------------------------------------------------
//! \fn  void CosmicRayAW::AssembleCRAWTasks
//! \brief Adds craw tasks to the task lists used by the time integrator.
//! Called by MeshBlockPack::AddPhysics() directly after the CosmicRayAW constructor.

void CosmicRayAW::AssembleCRAWTasks(
                              std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // "before_craw_sub" task list: post MPI receives for u before each subcycle
  id.irecv_sub = tl["before_craw_sub"]->AddTask(&CosmicRayAW::InitRecv, this, none);

  // "craw_sub" task list: one transport + source subcycle of size craw_dt_sub, run by the
  // driver N_cr/2 times before and N_cr/2 times after the fluid RK stages. No RK weights.
  id.flux    = tl["craw_sub"]->AddTask(&CosmicRayAW::CalculateAnisoFluxes, this, none);
  id.sendf   = tl["craw_sub"]->AddTask(&CosmicRayAW::SendFlux, this, id.flux);
  id.recvf   = tl["craw_sub"]->AddTask(&CosmicRayAW::RecvFlux, this, id.sendf);
  id.sub_rkupdt = tl["craw_sub"]->AddTask(&CosmicRayAW::Sub_RKUpdate, this, id.recvf);
  id.srctrms = tl["craw_sub"]->AddTask(&CosmicRayAW::CRAWSrcTerms, this, id.sub_rkupdt);
  id.restu     = tl["craw_sub"]->AddTask(&CosmicRayAW::RestrictU, this, id.srctrms);
  id.sendu_sub = tl["craw_sub"]->AddTask(&CosmicRayAW::SendU, this, id.restu);
  id.recvu_sub = tl["craw_sub"]->AddTask(&CosmicRayAW::RecvU, this, id.sendu_sub);
  id.prol_sub  = tl["craw_sub"]->AddTask(&CosmicRayAW::Prolongate, this, id.recvu_sub);
  id.bcs_sub   = tl["craw_sub"]->AddTask(&CosmicRayAW::ApplyPhysicalBCs, this, id.prol_sub);

  // "stagen" task list: the CR copy u0 -> u1 at stage 1, as MHD's CopyCons, then the
  // NewTimeStep that runs only on the last RK stage (gated inside), so it sets dtnew once
  // per cycle, not once per craw subcycle.
  // stagen: CR copy u0 -> u1 at stage 1. With MHD present it goes before MHD's flux task
  // and after MHD's own copy, so u1 is set before the MHD RK update that reads it. Inserting
  // ahead of flux moves flux's dependency onto this task, so it must also follow copyu.
  if (pmy_pack->pmhd != nullptr) {
    id.copyu = tl["stagen"]->InsertTask(&CosmicRayAW::CopyCons, this,
                                        pmy_pack->pmhd->id.copyu, pmy_pack->pmhd->id.flux);
  } else {
    id.copyu = tl["stagen"]->AddTask(&CosmicRayAW::CopyCons, this, none);
  }
  // CR adiabatic update in stagen, inserted between MHD's rkupdt and srctrms. It must follow
  // MHD's RK update, because it adds the gas-energy pdV increment to MHD's u0 without a stage
  // weight, and must precede the BCs and ConToPrim that read u0. Without MHD there is no face
  // solve, so there is no adiabatic CR update.
  TaskID cr_done = id.copyu;
  if (pmy_pack->pmhd != nullptr) {
    id.rkupdt = tl["stagen"]->InsertTask(&CosmicRayAW::RKUpdate, this,
                                         pmy_pack->pmhd->id.rkupdt, pmy_pack->pmhd->id.srctrms);
    cr_done = id.rkupdt;
  }
  // CR ghost exchange in stagen, every stage, after the CR update. Same order as MHD:
  // sendu -> recvu -> prol -> bcs.
  id.sendu = tl["stagen"]->AddTask(&CosmicRayAW::SendU, this, cr_done);
  id.recvu = tl["stagen"]->AddTask(&CosmicRayAW::RecvU, this, id.sendu);
  id.prol  = tl["stagen"]->AddTask(&CosmicRayAW::Prolongate, this, id.recvu);
  id.bcs   = tl["stagen"]->AddTask(&CosmicRayAW::ApplyPhysicalBCs, this, id.prol);
  id.newdt = tl["stagen"]->AddTask(&CosmicRayAW::NewTimeStep, this, none);

  // "after_craw_sub" task list: finish MPI sends and receives after each subcycle
  id.csend_sub = tl["after_craw_sub"]->AddTask(&CosmicRayAW::ClearSend, this, none);
  id.crecv_sub = tl["after_craw_sub"]->AddTask(&CosmicRayAW::ClearRecv, this, id.csend_sub);

  // "before_stagen": post receives for u before each stage, as MHD does
  id.irecv = tl["before_stagen"]->AddTask(&CosmicRayAW::InitRecv, this, none);

  // "after_stagen": finish sends and receives of u after each stage, as MHD does
  id.csend = tl["after_stagen"]->AddTask(&CosmicRayAW::ClearSend, this, none);
  id.crecv = tl["after_stagen"]->AddTask(&CosmicRayAW::ClearRecv, this, id.csend);

  return;
}

//----------------------------------------------------------------------------------------
// before_stagen

//! \fn TaskStatus CosmicRayAW::InitRecv
//! \brief Post MPI receives for u
TaskStatus CosmicRayAW::InitRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->InitRecv(nvars);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of u
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->InitFluxRecv(nvars);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
// stagen

//! \fn TaskStatus CosmicRayAW::CopyCons
//! \brief Copy u0 -> u1 at the start of the stage
TaskStatus CosmicRayAW::CopyCons(Driver *, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

//! \fn KOKKOS_INLINE_FUNCTION Real BnCell
//! \brief Unit projection b_n = B_x/|B| of the cell-centred field on the x1 direction.
//! Returns 1 when no MHD fluid is present (standalone CR runs keep the uniform field).
//! A cell with zero field returns 0, so it carries no anisotropic transport.
KOKKOS_INLINE_FUNCTION
Real BnCell(const DvceArray5D<Real> &bcc, const bool use_mhd, const int m, const int k,
            const int j, const int i) {
  if (!use_mhd) return 1.0;
  Real bx = bcc(m,IBX,k,j,i), by = bcc(m,IBY,k,j,i), bz = bcc(m,IBZ,k,j,i);
  Real bmag = std::sqrt(bx*bx + by*by + bz*bz);
  return (bmag > 0.0) ? bx/bmag : 0.0;
}

//! \fn KOKKOS_INLINE_FUNCTION Real BnFace
//! \brief Interface projection b_n = B_x,face / |B_side| on one side of an x1 face. The
//! normal component B_x is the constrained-transport face value, shared by both sides. The
//! magnitude |B| is taken from the cell-centred field on that side (cell ic = i-1 or i).
//! Standalone CR runs (no MHD) return 1.
KOKKOS_INLINE_FUNCTION
Real BnFace(const DvceArray5D<Real> &bcc, const DvceArray4D<Real> &bx1f, const bool use_mhd,
            const int m, const int k, const int j, const int i_face, const int ic) {
  if (!use_mhd) return 1.0;
  Real bx = bx1f(m,k,j,i_face);
  Real bxc = bcc(m,IBX,k,j,ic), byc = bcc(m,IBY,k,j,ic), bzc = bcc(m,IBZ,k,j,ic);
  Real bmag = std::sqrt(bxc*bxc + byc*byc + bzc*bzc);
  return (bmag > 0.0) ? bx/bmag : 0.0;
}

//! \fn TaskStatus CosmicRayAW::CalculateAnisoFluxes
//! TODO(roark): x2/x3 face fluxes for multi-D.
//! \brief Conservative anisotropic transport fluxes on x1 faces, written into uflx_aniso
//! by assignment. The projections b_nL, b_nR and the Alfven speeds are cell-centred values
//! from the MHD state; standalone CR runs use b_n = 1 and the input v_A.
//!   E_c flux through a face  = (b_n f_c) - S/2 [E_c],   S = v_max sqrt(gamma_cr-1) max|b_n|
//!   f_c slot stores the Riemann star value f_c*, not a flux. Sub_RKUpdate builds D^f from it.
//!   E_+- flux through a face = +-(b_n v_A E_+-) - S_a/2 [E_+-],  S_a = max |(b_n v_A)_{L,R}|
TaskStatus CosmicRayAW::CalculateAnisoFluxes(Driver *, int) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto fx1 = uflx_aniso.x1f;
  bool use_mhd = (pmy_pack->pmhd != nullptr);
  DvceArray5D<Real> bcc_ = use_mhd ? pmy_pack->pmhd->bcc0 : u0;
  DvceArray4D<Real> bx1f_ = use_mhd ? pmy_pack->pmhd->b0.x1f : DvceArray4D<Real>();
  DvceArray5D<Real> w0_  = use_mhd ? pmy_pack->pmhd->w0 : u0;
  Real vA_sa = v_A;
  
  // gas constant for CR pressure, P_c = (gamma_cr - 1) E_c
  Real gm1 = gamma_cr - 1.0;
  Real vmax = v_max;
  Real vmax2 = vmax*vmax;
  Real sqrt_gm1 = std::sqrt(gm1);

  // x1-faces: faces is..ie+1 for cells is..ie. Face i is the left face of cell i.
  par_for("craw_aniso_x1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // projections of b onto the x1 face from the cells i-1 (left) and i (right)
    Real bnL = BnFace(bcc_, bx1f_, use_mhd, m, k, j, i, i-1);
    Real bnR = BnFace(bcc_, bx1f_, use_mhd, m, k, j, i, i);
    Real bmax = fmax(fabs(bnL), fabs(bnR));

    // field does not pierce the face: no anisotropic transport through it.
    // This also avoids dividing by S below, which vanishes when bmax = 0.
    if (bmax == 0.0) {
      fx1(m,ICRE,k,j,i) = 0.0;
      fx1(m,ICRF,k,j,i) = 0.0;
      fx1(m,IAWP,k,j,i) = 0.0;
      fx1(m,IAWM,k,j,i) = 0.0;
    } else {
      // intermediate-state speed, per face: S = v_max sqrt(gamma_cr - 1) max(|b_nL|,|b_nR|)
      Real S = vmax*sqrt_gm1*bmax;

      
      fx1(m,ICRE,k,j,i) = (bnL * u0_(m,ICRF,k,j,i-1) + bnR * u0_(m,ICRF,k,j,i)) / 2.0 - S *(u0_(m,ICRE,k,j,i) - u0_(m,ICRE,k,j,i-1)) / 2.0;

      // Star state of the Riemann problem for (E_c, f_c). The f_c equation is
      // d_t f_c + S^2 d_x E_c = 0, so its physical face flux is S^2 * E_c*, not f_c*.
      // E_c* = (E_L+E_R)/2 + (f_L-f_R)/(2S)
      Real Ec_star = 0.5*(u0_(m,ICRE,k,j,i-1) + u0_(m,ICRE,k,j,i)) + (bnL*u0_(m,ICRF,k,j,i-1) - bnR*u0_(m,ICRF,k,j,i)) /(2.0*S);
      Real bLbar = (3*bnL + bnR)/4.0;
      Real bRbar = (3*bnR + bnL)/4.0;
      Real fc_star = 0.5*(u0_(m,ICRF,k,j,i-1) + u0_(m,ICRF,k,j,i)) 
                      - vmax2 * bLbar*gm1*(Ec_star- u0_(m,ICRE,k,j,i-1))/(2*S)
                      + vmax2 * bRbar*gm1*(Ec_star- u0_(m,ICRE,k,j,i))/(2*S);
      fx1(m,ICRF,k,j,i) = fc_star;

      // E_+: moves at +v_A, so its upwind cell is the one to the left of the face
      // b_n v_A = (B_x/|B|)(|B|/sqrt(rho)) = B_x/sqrt(rho): the |B| cancels, so use the face B_x
      Real pL = use_mhd ? bx1f_(m,k,j,i)/std::sqrt(w0_(m,IDN,k,j,i-1)) : vA_sa;
      Real pR = use_mhd ? bx1f_(m,k,j,i)/std::sqrt(w0_(m,IDN,k,j,i))   : vA_sa;
      Real Sa = fmax(fabs(pL), fabs(pR));
      fx1(m,IAWP,k,j,i) = (pL * u0_(m,IAWP,k,j,i-1) + pR * u0_(m,IAWP,k,j,i)) / 2.0
                          - Sa *(u0_(m,IAWP,k,j,i) - u0_(m,IAWP,k,j,i-1)) / 2.0;
      // E_-: moves at -v_A, so its upwind cell is the one to the right of the face
      fx1(m,IAWM,k,j,i) = -1*(pL * u0_(m,IAWM,k,j,i-1) + pR * u0_(m,IAWM,k,j,i)) / 2.0
                          - Sa *(u0_(m,IAWM,k,j,i) - u0_(m,IAWM,k,j,i-1)) / 2.0;
      
    }
  });

  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::SendFlux
//! \brief Pack and send restricted fluxes of u to coarse neighbours (SMR/AMR only)
TaskStatus CosmicRayAW::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->PackAndSendFluxCC(uflx);
  }
  return tstat;
}

//! \fn TaskStatus CosmicRayAW::RecvFlux
//! \brief Receive and unpack restricted fluxes of u from fine neighbours (SMR/AMR only)
TaskStatus CosmicRayAW::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  }
  return tstat;
}

//! \fn TaskStatus CosmicRayAW::Sub_RKUpdate
//! \brief Forward subcycle update with the flux divergence and the non-conservative f_c term.
//! Wave speed per face and the smooth coefficient come from the cell-centred field:
//!   S_{i+1/2} = v_max sqrt(gamma_cr-1) max(|b_i|, |b_{i+1}|),   H_i = (gamma_cr-1) v_max^2 b_i
//! TODO(roark): x2/x3 flux divergence for multi-D.
TaskStatus CosmicRayAW::Sub_RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int ks = indcs.ks, ke = indcs.ke;
  int js = indcs.js, je = indcs.je;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);

  // forward subcycle step: u <- u - dt_sub*divF (+ non-conservative term), no RK weights
  Real beta_dt = pdriver->craw_dt_sub;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto flx1 = uflx_aniso.x1f;
  auto &mbsize = pmy_pack->pmb->mb_size;
  CREOS_Data creos_ = pcreos->eos_data;
  Real gm1 = gamma_cr - 1.0;
  Real vmax = v_max;
  Real vmax2 = vmax*vmax;
  Real sqrt_gm1 = std::sqrt(gm1);
  bool use_mhd = (pmy_pack->pmhd != nullptr);
  DvceArray5D<Real> bcc_ = use_mhd ? pmy_pack->pmhd->bcc0 : u0;
  DvceArray4D<Real> bx1f_ = use_mhd ? pmy_pack->pmhd->b0.x1f : DvceArray4D<Real>();
  DvceArray5D<Real> mu0_ = use_mhd ? pmy_pack->pmhd->u0 : u0;  // gas momentum, MHD u0

  // 1D (x1) update, same weights as hydro's RKUpdate. Loops over all nvars components.
  // This is the anisotropic subcycle update, not the adiabatic RKUpdate in stagen.
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("craw_update",DevExeSpace(),scr_size,scr_level,0,nmb1,0,nvars-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
    ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

    if (n == ICRF) {
      // non-conservative f_c term: linear fluctuations (path-conservative scheme).
      // flx1 holds the star value f_c* at each face. The cell's own state is f_i.
      //   D_L at right face i+1/2 = +S_{i+1/2} (f_i - f*_{i+1/2})
      //   D_R at left face  i-1/2 = -S_{i-1/2} (f_i - f*_{i-1/2})
      // smooth term H_i d_x E_c with H_i = (gamma_cr-1) v_max^2 b_i and a central difference
      // of E_c on the cell stencil.
      // divf = (D_L^{i+1/2} - D_R^{i-1/2}) / dx + H_i d_x E_c
      par_for_inner(member, is, ie, [&](const int i) {
        // shared face B_x on faces i (left of cell i) and i+1 (right of cell i);
        // each face's wave speed is the max |b_n| of its two sides
        Real bnf_Lm = BnFace(bcc_, bx1f_, use_mhd, m, k, j, i,   i-1);
        Real bnf_Lr = BnFace(bcc_, bx1f_, use_mhd, m, k, j, i,   i);
        Real bnf_Rl = BnFace(bcc_, bx1f_, use_mhd, m, k, j, i+1, i);
        Real bnf_Rr = BnFace(bcc_, bx1f_, use_mhd, m, k, j, i+1, i+1);
        Real S_L = vmax*sqrt_gm1*fmax(fabs(bnf_Lm), fabs(bnf_Lr));
        Real S_R = vmax*sqrt_gm1*fmax(fabs(bnf_Rl), fabs(bnf_Rr));
        // cell midpoint projection b_i = B_x,i/|B_i| for H_i (paper's H_i = (gamma-1) c^2 b_i)
        Real bi = BnCell(bcc_, use_mhd, m, k, j, i);
        Real fc_star_ip1 = flx1(m,n,k,j,i+1);
        Real fc_star_i   = flx1(m,n,k,j,i);
        Real DL_right = S_R*(u0_(m,n,k,j,i) - fc_star_ip1);
        Real DR_left  = -S_L*(u0_(m,n,k,j,i) - fc_star_i);
        Real dx1 = mbsize.d_view(m).dx1;
        Real dEc_dx = (u0_(m,ICRE,k,j,i+1) - u0_(m,ICRE,k,j,i-1))/(2.0*dx1);
        Real Hsmooth = gm1*vmax2*bi*dEc_dx;
        divf(i) = (DL_right - DR_left)/dx1 + Hsmooth;
        // Gas x-momentum from the same f_cr update. Paper Eq. 330: D^{rho u} = -(b/c^2) D^f,
        // so Delta(rho u) = -(b/c^2) Delta f with Delta f = -beta_dt*divf. Only the ICRF team
        // writes IM1 and IEN for this cell, so there is no race.
        // The kick changes kinetic energy, not thermal energy (paper Eqs. 344-346, 416-418):
        // IEN += (m_new^2 - m_old^2)/(2 rho). rho is the fluid density, unchanged by craw.
        // Over the subcycles these increments telescope to (m_N^2 - m_0^2)/(2 rho), the paper's
        // total kinetic change for the source step.
        if (use_mhd) {
          Real rho  = mu0_(m,IDN,k,j,i);
          Real m0   = mu0_(m,IM1,k,j,i);
          Real m1   = m0 + beta_dt*bi*divf(i)/vmax2;
          mu0_(m,IM1,k,j,i) = m1;
          mu0_(m,IEN,k,j,i) += (m1*m1 - m0*m0)/(2.0*rho);
        }
      });
    } else {
      // conservative flux difference for ICRE, IAWP, IAWM
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
      });
    }
    member.team_barrier();

    par_for_inner(member, is, ie, [&](const int i) {
      Real unew = u0_(m,n,k,j,i) - beta_dt*divf(i);
      // floor the positive-definite energies; f_c (ICRF) is signed and is not floored
      u0_(m,n,k,j,i) = creos_.FloorSlot(n, unew);
    });
  });

  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::CRAWSrcTerms
//! \brief TODO(roark): growth/damping source terms for E_c and E_+-, equal and opposite.
//! Per-cell implicit solve of (I - dt R) u_new = u_old, with R from the anisotropic
//! source. Block structure: (E_c, f_c) solved as a 2x2, then E_+ and E_- as scalars.
TaskStatus CosmicRayAW::CRAWSrcTerms(Driver *pdrive, int stage) {
  Real dt_src = pdrive->craw_dt_sub;  // source span of one craw subcycle

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // standalone coefficients (constant) and the MHD state the per-cell ones come from
  Real v_const = v_A;
  Real x_const = chi_cr;
  Real alpha_const = alpha_landau;
  Real c = v_max;
  Real y = gamma_cr;
  bool use_mhd = (pmy_pack->pmhd != nullptr);
  DvceArray5D<Real> w0_ = use_mhd ? pmy_pack->pmhd->w0 : u0;
  DvceArray5D<Real> mu0_ = use_mhd ? pmy_pack->pmhd->u0 : u0;  // gas energy, MHD u0
  DvceArray5D<Real> bcc_ = use_mhd ? pmy_pack->pmhd->bcc0 : u0;
  Real gam_fl = use_mhd ? pmy_pack->pmhd->peos->eos_data.gamma : 1.0;
  Real chiK = chi_K, alphaK = alpha_K, Eu = E_unit, vu = v_unit, Tu = T_unit;
  Real Bfac = std::sqrt(4.0*M_PI*E_unit);  // B_gauss = B_HL * Bfac, B_HL the field in code units
  Real atol_ = atol;
  Real rtol_ = rtol;
  CREOS_Data creos_ = pcreos->eos_data;
  Real gs = 1.0 - 1.0/std::sqrt(2.0);  // gamma_s of the two-stage scheme
  const int max_src_iter = 5000000;      // per-cell cap on adaptive substeps
  auto u0_ = u0;
  // first failing cell, as a flat index (-1 = none). Only the first failure is recorded.
  DvceArray1D<int> fail_idx("craw_src_fail", 1);
  Kokkos::deep_copy(DevExeSpace(), fail_idx, -1);
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;

  par_for("craw_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // per-cell coefficients. Standalone: the constants. MHD: from the local state at the start
    // of this source step, so the temperature used for alpha is the start-of-step value.
    Real v = v_const, x = x_const, alpha = alpha_const;
    if (use_mhd) {
      Real rho = w0_(m,IDN,k,j,i);
      Real p = (gam_fl - 1.0)*w0_(m,IEN,k,j,i);  // IEN of the primitive state is eint
      Real bx = bcc_(m,IBX,k,j,i), by = bcc_(m,IBY,k,j,i), bz = bcc_(m,IBZ,k,j,i);
      Real bhl = sqrt(bx*bx + by*by + bz*bz);
      Real bg = bhl*Bfac;                              // field in gauss
      v = bhl/sqrt(rho);                               // v_A = B/sqrt(rho), code units
      if (bg > 0.0) {
        x = chiK/bg;                                   // chi = chi_K / B
        // sqrt(kT/m_p) = sqrt(P/rho) in cgs = sqrt(P/rho)_code * v_unit
        alpha = (alphaK/bg)*sqrt(p/rho)*vu*Eu*Tu;
      } else {
        x = 0.0;                                       // no field: no gyroresonance, no alpha
        alpha = 0.0;
      }
    }

    // Un: stage-start state for this cell
    Real Un[4] = {u0_(m,ICRE,k,j,i), u0_(m,ICRF,k,j,i),
                  u0_(m,IAWP,k,j,i), u0_(m,IAWM,k,j,i)};

    // R[V] * V for a state V, and the block solve (I - c*dt*R[V]) X = rhs.
    // The 4x4 R has the block structure of the matrix in the scheme: (E_c,f_c) rows
    // depend only on (E_c,f_c) and the frozen E_+-; E_+- rows are diagonal in E_+-.
    auto solve = [&](const Real *V, Real cdt, const Real *rhs, Real *X) {
      Real Ec = V[0], fc = V[1], Ep = V[2], Em = V[3];
      Real R11 = v*v*x*y*(Ep+Em),  R12 = -v*x*(Ep-Em);
      Real R21 = c*c*v*x*y*(Ep-Em), R22 = -c*c*x*(Ep+Em);
      Real R31 = -v*v*x*y*Ep, R32 = v*x*Ep, R33 = -(alpha*Ep);
      Real R41 = -v*v*x*y*Em, R42 = -v*x*Em, R44 = -(alpha*Em);
      (void)Ec; (void)fc;

      // 2x2 block for (E_c, f_c)
      Real a11 = 1.0 - cdt*R11, a12 = -cdt*R12;
      Real a21 = -cdt*R21,      a22 = 1.0 - cdt*R22;
      Real det = a11*a22 - a12*a21;
      X[0] = ( a22*rhs[0] - a12*rhs[1])/det;
      X[1] = (-a21*rhs[0] + a11*rhs[1])/det;

      // scalar solves for E_+ and E_-, with the (E_c, f_c) result moved to the right side
      X[2] = (rhs[2] + cdt*(R31*X[0] + R32*X[1]))/(1.0 - cdt*R33);
      X[3] = (rhs[3] + cdt*(R41*X[0] + R42*X[1]))/(1.0 - cdt*R44);
    };

    // R[V]*W for a vector W (used in the explicit parts of the final stage)
    auto Rmul = [&](const Real *V, const Real *W, Real *out) {
      Real Ec = V[0], fc = V[1], Ep = V[2], Em = V[3];
      (void)Ec; (void)fc;
      Real R11 = v*v*x*y*(Ep+Em),  R12 = -v*x*(Ep-Em);
      Real R21 = c*c*v*x*y*(Ep-Em), R22 = -c*c*x*(Ep+Em);
      Real R31 = -v*v*x*y*Ep, R32 = v*x*Ep, R33 = -(alpha*Ep);
      Real R41 = -v*v*x*y*Em, R42 = -v*x*Em, R44 = -(alpha*Em);
      out[0] = R11*W[0] + R12*W[1];
      out[1] = R21*W[0] + R22*W[1];
      out[2] = R31*W[0] + R32*W[1] + R33*W[2];
      out[3] = R41*W[0] + R42*W[1] + R44*W[3];
    };

    // subcycle this cell from t = 0 to dt_src with its own adaptive step
    Real U[4] = {Un[0], Un[1], Un[2], Un[3]};
    Real t = 0.0;
    Real dtsrc = dt_src;  // first trial step is the full stage interval
    int niter = 0;
    while (t < dt_src) {
      if (++niter > max_src_iter) {
        // record this cell as the failure, only if none has been recorded yet
        int flat = ((m*ncells3 + k)*ncells2 + j)*ncells1 + i;
        Kokkos::atomic_compare_exchange(&fail_idx(0), -1, flat);
        break;
      }
      if (t + dtsrc > dt_src) dtsrc = dt_src - t;  // do not step past the stage end

      // 1. predictor: (I - dt R[U]) Up = U
      Real Up[4];
      solve(U, dtsrc, U, Up);

      // 2. midpoint state Ur = (U + Up)/2
      Real Ur[4];
      for (int n = 0; n < 4; ++n) Ur[n] = 0.5*(U[n] + Up[n]);

      // 3. corrector: (I - gs*dt R[Ur]) U* = U
      Real Ustar[4];
      solve(Ur, gs*dtsrc, U, Ustar);

      // 4. final: (I - gs*dt R[Ur]) Un+1 = U + (1-gs) dt R[Ur] U*
      Real RU[4];
      Rmul(Ur, Ustar, RU);
      Real rhs_f[4];
      for (int n = 0; n < 4; ++n) rhs_f[n] = U[n] + (1.0-gs)*dtsrc*RU[n];
      Real Unp1[4];
      solve(Ur, gs*dtsrc, rhs_f, Unp1);

      // error: max over the four variables of |Up - Un+1| / (atol + rtol max(|Up|,|Un+1|))
      Real err = 0.0;
      for (int n = 0; n < 4; ++n) {
        Real scale = atol_ + rtol_*fmax(fabs(Up[n]), fabs(Unp1[n]));
        err = fmax(err, fabs(Up[n] - Unp1[n])/scale);
      }

      // controller, standard form: factor = 0.9/sqrt(err)
      Real fac = fmin(5.0, 0.9/sqrt(err));
      if (err <= 1.0) {
        // accept: advance time and state, then set the next step
        t += dtsrc;
        for (int n = 0; n < 4; ++n) U[n] = Unp1[n];
      }
      // reject: keep U and retry from the same t with the new step
      dtsrc = dtsrc*fac;
    }

    // floors on the positive-definite energies. f_c is signed and is not floored.
    Real Ec_new  = creos_.FloorSlot(ICRE, U[0]);
    Real Ea_p_new = creos_.FloorSlot(IAWP, U[2]);
    Real Ea_m_new = creos_.FloorSlot(IAWM, U[3]);
    u0_(m,ICRE,k,j,i) = Ec_new;
    u0_(m,ICRF,k,j,i) = U[1];
    u0_(m,IAWP,k,j,i) = Ea_p_new;
    u0_(m,IAWM,k,j,i) = Ea_m_new;

    // Thermal energy transfer over the source step (paper Sec. 3.3, Eq. 416):
    //   Delta(eps_th) = -Delta(eps_cr + eps_a+ + eps_a-)
    // Uses the floored final values, so the gas gains exactly what the wave and CR energy
    // lost, including any floor injection. Written once per cell by this thread. The kinetic
    // energy change is separate (Sub_RKUpdate), so the total gas IEN change here is thermal.
    // Landau heating is part of the wave-energy loss and is included automatically.
    if (use_mhd) {
      Real dE = (Ec_new - Un[0]) + (Ea_p_new - Un[2]) + (Ea_m_new - Un[3]);
      mu0_(m,IEN,k,j,i) -= dE;
    }
  });

  // host check: stop the run as soon as any cell has exceeded the iteration cap
  auto fail_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), fail_idx);
  if (fail_h(0) >= 0) {
    int flat = fail_h(0);
    int i = flat % ncells1;
    int j = (flat / ncells1) % ncells2;
    int k = (flat / (ncells1*ncells2)) % ncells3;
    int m = flat / (ncells1*ncells2*ncells3);
    std::cout << "COSMIC RAY FUCKED UP: source subcycle exceeded " << max_src_iter
              << " iterations at m=" << m << " k=" << k << " j=" << j << " i=" << i
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::RestrictU
//! \brief Restrict u to coarse_u0 for SMR/AMR (only with multilevel grids)
TaskStatus CosmicRayAW::RestrictU(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::SendU
//! \brief Pack and send cell-centered u boundary data
TaskStatus CosmicRayAW::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//! \fn TaskStatus CosmicRayAW::RecvU
//! \brief Receive and unpack cell-centered u boundary data
TaskStatus CosmicRayAW::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//! \fn TaskStatus CosmicRayAW::ApplyPhysicalBCs
//! \brief Apply physical boundary conditions to u.
//! Outflow (zero-gradient) matches hydro, MHD, and radiation: each ghost zone copies the
//! nearest interior cell. Reflect and inflow are not implemented yet, because which CR
//! components change sign or take inflow values is a physics decision.
TaskStatus CosmicRayAW::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto &pm = pmy_pack->pmesh;
  auto u0_ = u0;

  // x1 faces
  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    par_for("crawbc_x1", DevExeSpace(), 0,(nmb-1),0,(nvars-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::outflow) {
        for (int i=0; i<ng; ++i) {
          u0_(m,n,k,j,is-i-1) = u0_(m,n,k,j,is);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::outflow) {
        for (int i=0; i<ng; ++i) {
          u0_(m,n,k,j,ie+i+1) = u0_(m,n,k,j,ie);
        }
      }
    });
  }

  // x2 faces
  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    par_for("crawbc_x2", DevExeSpace(), 0,(nmb-1),0,(nvars-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::outflow) {
        for (int j=0; j<ng; ++j) {
          u0_(m,n,k,js-j-1,i) = u0_(m,n,k,js,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::outflow) {
        for (int j=0; j<ng; ++j) {
          u0_(m,n,k,je+j+1,i) = u0_(m,n,k,je,i);
        }
      }
    });
  }

  // x3 faces
  if (pm->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
    par_for("crawbc_x3", DevExeSpace(), 0,(nmb-1),0,(nvars-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::outflow) {
        for (int k=0; k<ng; ++k) {
          u0_(m,n,ks-k-1,j,i) = u0_(m,n,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::outflow) {
        for (int k=0; k<ng; ++k) {
          u0_(m,n,ke+k+1,j,i) = u0_(m,n,ke,j,i);
        }
      }
    });
  }

  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::Prolongate
//! \brief Prolongate u at fine/coarse boundaries with SMR/AMR
TaskStatus CosmicRayAW::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    pbval_u->ProlongateCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus CosmicRayAW::RKUpdate
//! \brief Adiabatic CR update for one RK stage, in stagen after MHD's flux task, so that
//! uface_cr (contact velocity u*) and crflx_adv (upwinded CR fluxes) are current.
//! Same stage weights as MHD's update, with u1 holding the stage-start state:
//!   u0 = gam0*u0 + gam1*u1 + beta_dt*(-divF + S)
//!   divF = (crflx_adv(i+1) - crflx_adv(i))/dx,   divu = (u*(i+1) - u*(i))/dx
//!   S: ICRE -> -P_cr divu;  ICRF -> -f_cr b_x^2 divu;  IAWP, IAWM -> -P_a divu
//! The gas-energy pdV term (P_cr + P_a) divu is not here: it belongs to the fluid energy.
TaskStatus CosmicRayAW::RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  Real gam0 = pdriver->gam0[stage-1];
  Real gam1 = pdriver->gam1[stage-1];
  Real beta_dt = pdriver->beta[stage-1]*pmy_pack->pmesh->dt;

  bool use_mhd = (pmy_pack->pmhd != nullptr);
  DvceArray5D<Real> bcc_ = use_mhd ? pmy_pack->pmhd->bcc0 : u0;
  auto u0_ = u0;
  auto u1_ = u1;
  auto mu0_ = use_mhd ? pmy_pack->pmhd->u0 : u0;  // MHD u0, for the gas-energy pdV increment
  auto crf_ = crflx_adv.x1f;
  auto ust_ = uface_cr.x1f;
  auto &mbsize = pmy_pack->pmb->mb_size;
  CREOS_Data creos_ = pcreos->eos_data;

  // One thread per cell. It updates every CR slot and writes the gas IEN once, so no atomic.
  // Each slot's pressure-work part is -beta_dt*src, and the gas gets that after the floor:
  //   gas IEN += sum over energy slots (flux_part - floored new value) = -beta_dt*sum(src)
  // f_cr (ICRF) is a flux, not an energy, so it has no gas term.
  // All reads of u0/u1/crflx/uface are from values this thread does not overwrite, except its
  // own u0 slots, which it reads before writing.
  par_for("craw_rkupdt", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = mbsize.d_view(m).dx1;
    Real divu = (ust_(m,k,j,i+1) - ust_(m,k,j,i))/dx1;
    // Pressures come from the stage-start state u1, which this kernel never writes.
    Real Pcr = creos_.CRPressure(u1_(m,ICRE,k,j,i));
    Real bn = BnCell(bcc_, use_mhd, m, k, j, i);  // b_x, 1 without MHD
    Real gas_gain = 0.0;
    for (int n = 0; n < nvars; ++n) {
      Real divF = (crf_(m,n,k,j,i+1) - crf_(m,n,k,j,i))/dx1;
      Real u_old = u0_(m,n,k,j,i);
      Real flux_part = gam0*u_old + gam1*u1_(m,n,k,j,i) - beta_dt*divF;
      Real src = 0.0;
      if (n == ICRE) {
        src = -Pcr*divu;
      } else if (n == ICRF) {
        src = -u_old*bn*bn*divu;
      } else {
        src = -creos_.WavePressure(u1_(m,n,k,j,i))*divu;
      }
      Real ufloor = creos_.FloorSlot(n, flux_part + beta_dt*src);
      u0_(m,n,k,j,i) = ufloor;
      if (use_mhd && n != ICRF) {
        gas_gain += flux_part - ufloor;
      }
    }
    if (use_mhd) {
      mu0_(m,IEN,k,j,i) += gas_gain;
    }
  });
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::NewTimeStep
//! TODO(roark): include b_n in S once the field is read from the mesh.
//! \brief Sets dtnew from the anisotropic CFL limit dx1/v_max. Computed on the last stage
//! only, as in hydro. Mesh::NewTimeStep applies the CFL factor and takes the minimum.
TaskStatus CosmicRayAW::NewTimeStep(Driver *pdrive, int stage) {
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete;  // only execute last stage
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nmkji = (pmy_pack->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;

  // wave speed captured by value for the kernel
  Real S = v_max*std::sqrt(gamma_cr - 1.0);
  auto &mbsize = pmy_pack->pmb->mb_size;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;

  // Fluid limit with the cosmic-ray pressure in the fast speed. Uses the effective pressure
  // p_eff = P_th + (gamma_cr P_cr + gamma_a P_a)/gamma_th, so gamma_th p_eff is the summed
  // pressure-weighted sound speed. Only the Newtonian ideal-gas MHD case is covered here,
  // which is the case the paper's test problems use. The fluid signal speed is |v| + c_f.
  bool use_mhd = (pmy_pack->pmhd != nullptr);
  DvceArray5D<Real> w0_ = use_mhd ? pmy_pack->pmhd->w0 : u0;
  DvceArray5D<Real> bcc_ = use_mhd ? pmy_pack->pmhd->bcc0 : u0;
  EOS_Data eos = use_mhd ? pmy_pack->pmhd->peos->eos_data : EOS_Data();
  CREOS_Data creos_ = pcreos->eos_data;
  Real gcr = gamma_cr;
  Real gaw = creos_.gamma_a;
  auto u0_ = u0;  // CR state for the CR pressure in the fast speed
  int isI = indcs.is, jsI = indcs.js, ksI = indcs.ks;
  int nx1 = indcs.nx1;

  // Full-step limit, paper Eqs. 354-357: dt = min(N_cr*dt_cr,req, dt_mhd), written as one
  // speed per direction: dt = dx / max(S/N_cr, |v_d| + c_f,d).
  //   S/N_cr: the CR limit N_cr*dx/S restated as a speed, so it competes directly with the
  //           fluid speeds.
  //   c_f,d:  fluid fast speed with the CR and wave pressure in the effective sound speed.
  // The driver then runs N_cr subcycles of size dt/N_cr. Mesh applies the CFL factor.
  // Without MHD only S/N_cr remains.
  Real Sn = S/static_cast<Real>(N_cr);
  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();
  Kokkos::parallel_reduce("CRAWNewDt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/(nx2*nx1) + ksI;
    int j = ((idx - m*nkji)/nx1) % nx2 + jsI;
    int i = (idx - m*nkji) % nx1 + isI;
    Real s1 = Sn, s2 = Sn, s3 = Sn;
    if (use_mhd && eos.is_ideal) {
      Real d = w0_(m,IDN,k,j,i);
      Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
      Real Pcr = (gcr - 1.0)*u0_(m,ICRE,k,j,i);
      Real Pa  = (gaw - 1.0)*(u0_(m,IAWP,k,j,i) + u0_(m,IAWM,k,j,i));
      // p_eff = P_th + (gamma_cr P_cr + gamma_a P_a)/gamma_th, so gamma_th p_eff = sum_k gamma_k P_k
      p += (gcr*Pcr + gaw*Pa)/eos.gamma;
      Real bx = bcc_(m,IBX,k,j,i), by = bcc_(m,IBY,k,j,i), bz = bcc_(m,IBZ,k,j,i);
      Real cf1 = eos.IdealMHDFastSpeed(d, p, bx, by, bz);
      Real cf2 = eos.IdealMHDFastSpeed(d, p, by, bz, bx);
      Real cf3 = eos.IdealMHDFastSpeed(d, p, bz, bx, by);
      s1 = fmax(Sn, fabs(w0_(m,IVX,k,j,i)) + cf1);
      s2 = fmax(Sn, fabs(w0_(m,IVY,k,j,i)) + cf2);
      s3 = fmax(Sn, fabs(w0_(m,IVZ,k,j,i)) + cf3);
    }
    min_dt1 = fmin((mbsize.d_view(m).dx1/s1), min_dt1);
    min_dt2 = fmin((mbsize.d_view(m).dx2/s2), min_dt2);
    min_dt3 = fmin((mbsize.d_view(m).dx3/s3), min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3));

  // combine only the directions that are active, as hydro does
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// after_stagen

//! \fn TaskStatus CosmicRayAW::ClearSend
//! \brief Check that all MPI sends for u have completed
TaskStatus CosmicRayAW::ClearSend(Driver *pdrive, int stage) {
  // check sends of u complete
  // stage = -1 (ICs) clears the u sends; stage >= 0 clears them every stage
  TaskStatus tstat = TaskStatus::complete;
  if ((stage >= 0) || (stage == -1)) {
    tstat = pbval_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check sends of restricted fluxes of u complete
  // do not check flux send for ICs (stage < 0)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxSend();
  }
  return tstat;
}

//! \fn TaskStatus CosmicRayAW::ClearRecv
//! \brief Check that all MPI receives for u have completed
TaskStatus CosmicRayAW::ClearRecv(Driver *pdrive, int stage) {
  // check receives of u complete
  // stage = -1 (ICs) clears the u receives; stage >= 0 clears them every stage
  TaskStatus tstat = TaskStatus::complete;
  if ((stage >= 0) || (stage == -1)) {
    tstat = pbval_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check receives of restricted fluxes of u complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxRecv();
  }
  return tstat;
}

} // namespace craw
