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

  // "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&CosmicRayAW::InitRecv, this, none);

  // "stagen" task list
  // The CR flux is computed alongside the fluid flux, from the same stage-start state
  // (b0, w0) that the fluid Riemann solver reads. It must run before the fluid CT task,
  // which overwrites b0 for this stage. Dependencies are taken from the fluid task IDs,
  // which exist because the fluid assembly runs before AssembleCRAWTasks.
  id.copyu   = tl["stagen"]->AddTask(&CosmicRayAW::CopyCons, this, none);
  // CalculateAnisoFluxes needs both the fluid's stage-start state and this module's own
  // CopyCons, so it waits on both. Dependencies combine as a bitfield with operator|.
  TaskID fluid_dep = CRAWFluidFluxDependency(id.copyu);
  TaskID flux_dep = fluid_dep | id.copyu;
  id.flux    = tl["stagen"]->AddTask(&CosmicRayAW::CalculateAnisoFluxes, this, flux_dep);
  id.sendf   = tl["stagen"]->AddTask(&CosmicRayAW::SendFlux, this, id.flux);
  id.recvf   = tl["stagen"]->AddTask(&CosmicRayAW::RecvFlux, this, id.sendf);
  id.rkupdt  = tl["stagen"]->AddTask(&CosmicRayAW::RKUpdate, this, id.recvf);
  id.srctrms = tl["stagen"]->AddTask(&CosmicRayAW::CRAWSrcTerms, this, id.rkupdt);
  id.restu   = tl["stagen"]->AddTask(&CosmicRayAW::RestrictU, this, id.srctrms);
  id.sendu   = tl["stagen"]->AddTask(&CosmicRayAW::SendU, this, id.restu);
  id.recvu   = tl["stagen"]->AddTask(&CosmicRayAW::RecvU, this, id.sendu);
  id.prol    = tl["stagen"]->AddTask(&CosmicRayAW::Prolongate, this, id.recvu);
  id.bcs     = tl["stagen"]->AddTask(&CosmicRayAW::ApplyPhysicalBCs, this, id.prol);
  id.newdt   = tl["stagen"]->AddTask(&CosmicRayAW::NewTimeStep, this, id.bcs);

  // "after_stagen" task list
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

//! \fn TaskStatus CosmicRayAW::CalculateAnisoFluxes
//! TODO(roark): x2/x3 face fluxes for multi-D; b_nL/b_nR from the field; varying-b S.
//! \brief Conservative anisotropic transport fluxes for E_c and E_+-, written into
//! uflx_aniso by assignment (no zeroing needed). Uniform field along x1 (b_n = 1):
//!   E_c flux through a face  = b_n * f_c            (upwind on the sign of f_c)
//!   E_+- flux through a face = +-b_n * v_A * E_+-   (upwind on the sign of the wave)
//! The f_c face flux is S^2 E_c* with the Riemann star value E_c*, which is the
//! conservative flux of the f_c equation for a uniform field.
TaskStatus CosmicRayAW::CalculateAnisoFluxes(Driver *, int) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto fx1 = uflx_aniso.x1f;
  
  // gas constant for CR pressure, P_c = (gamma_cr - 1) E_c
  Real gm1 = gamma_cr - 1.0;
  Real vmax = v_max;
  Real vmax2 = vmax*vmax;
  Real vA = v_A;
  Real sqrt_gm1 = std::sqrt(gm1);

  // x1-faces: faces il..iu+1 for cells is..ie. Face i is the left face of cell i.
  par_for("craw_aniso_x1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // projections of b onto the x1 face from the left and right cells.
    // TODO(roark): replace these with b.x1 of the field in cells i-1 and i. Uniform
    // field along x1 for now, so both are 1.
    Real bnL = 1.0;
    Real bnR = 1.0;
    Real bmax = (bnL >bnR ? bnL : bnR);

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
      fx1(m,ICRF,k,j,i) = S*fc_star;

      // E_+: moves at +v_A, so its upwind cell is the one to the left of the face
      Real Sa = (vA*bnL > vA*bnR ? vA*bnL : vA*bnR);
      fx1(m,IAWP,k,j,i) = (bnL * vA * u0_(m,IAWP,k,j,i-1) + bnR * vA * u0_(m,IAWP,k,j,i)) / 2.0
                          - Sa *(u0_(m,IAWP,k,j,i) - u0_(m,IAWP,k,j,i-1)) / 2.0;
      // E_-: moves at -v_A, so its upwind cell is the one to the right of the face
      fx1(m,IAWM,k,j,i) = -1*(bnL * vA * u0_(m,IAWM,k,j,i-1) + bnR * vA * u0_(m,IAWM,k,j,i)) / 2.0
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

//! \fn TaskStatus CosmicRayAW::RKUpdate
//! \brief TODO(roark): x2/x3 flux divergence for multi-D.
TaskStatus CosmicRayAW::RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int ks = indcs.ks, ke = indcs.ke;
  int js = indcs.js, je = indcs.je;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto u1_ = u1;
  auto flx1 = uflx_aniso.x1f;
  auto &mbsize = pmy_pack->pmb->mb_size;
  Real S_cr = v_max*std::sqrt(gamma_cr - 1.0);  // uniform-field wave speed S
  Real bi_x1 = 1.0;  // b_n of the cell along x1; uniform field for now (TODO: from B)

  // 1D (x1) update, same weights as hydro's RKUpdate. Loops over all nvars components.
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("craw_update",DevExeSpace(),scr_size,scr_level,0,nmb1,0,nvars-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
    ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

    if (n == ICRF) {
      // non-conservative f_c term: linear fluctuations (path-conservative scheme).
      // flx1 holds S*fc* at each face, so fc* = flx1/S_cr. The cell's own state is f_i.
      //   D_L at right face i+1/2 = +S (f_i - fc*_{i+1/2})
      //   D_R at left face  i-1/2 = -S (f_i - fc*_{i-1/2})
      // smooth non-conservative term H_i d_x U|smooth, f_cr row: (gamma_cr-1) v_max^2 b d_x E_c
      // with a central difference of E_c on the cell stencil.
      // divf = (D_L^{i+1/2} - D_R^{i-1/2}) / dx + H_i d_x U|smooth
      par_for_inner(member, is, ie, [&](const int i) {
        Real fc_star_ip1 = flx1(m,n,k,j,i+1)/S_cr;
        Real fc_star_i   = flx1(m,n,k,j,i)/S_cr;
        Real DL_right = S_cr*(u0_(m,n,k,j,i) - fc_star_ip1);
        Real DR_left  = -S_cr*(u0_(m,n,k,j,i) - fc_star_i);
        Real dx1 = mbsize.d_view(m).dx1;
        Real dEc_dx = (u0_(m,ICRE,k,j,i+1) - u0_(m,ICRE,k,j,i-1))/(2.0*dx1);
        Real Hsmooth = (gamma_cr - 1.0)*v_max*v_max*bi_x1*dEc_dx;
        divf(i) = (DL_right - DR_left)/dx1 + Hsmooth;
      });
    } else {
      // conservative flux difference for ICRE, IAWP, IAWM
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
      });
    }
    member.team_barrier();

    par_for_inner(member, is, ie, [&](const int i) {
      u0_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - beta_dt*divf(i);
    });
  });

  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::CRAWSrcTerms
//! \brief TODO(roark): growth/damping source terms for E_c and E_+-, equal and opposite.
//! Per-cell implicit solve of (I - dt R) u_new = u_old, with R from the anisotropic
//! source. Block structure: (E_c, f_c) solved as a 2x2, then E_+ and E_- as scalars.
TaskStatus CosmicRayAW::CRAWSrcTerms(Driver *pdrive, int stage) {
  Real dt_src = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // parameters captured by value for the kernel
  Real v = v_A;
  Real c = v_max;
  Real y = gamma_cr;
  Real x = chi_cr;
  Real alpha = alpha_landau;
  Real atol_ = atol;
  Real rtol_ = rtol;
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

    u0_(m,ICRE,k,j,i) = U[0];
    u0_(m,ICRF,k,j,i) = U[1];
    u0_(m,IAWP,k,j,i) = U[2];
    u0_(m,IAWM,k,j,i) = U[3];
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

  // smallest dx/S in each direction, as in hydro: each direction is reduced separately
  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();
  Kokkos::parallel_reduce("CRAWNewDt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
    int m = (idx)/nkji;  // cell spacing depends only on the MeshBlock
    min_dt1 = fmin((mbsize.d_view(m).dx1/S), min_dt1);
    min_dt2 = fmin((mbsize.d_view(m).dx2/S), min_dt2);
    min_dt3 = fmin((mbsize.d_view(m).dx3/S), min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3));

  // combine only the directions that are active, as hydro does
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }
  // the CR limit on the full step: N_cr substeps of size dt_cr each satisfy dt_cr <= dt_cr_req
  dtnew = static_cast<Real>(N_cr)*dtnew;
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
