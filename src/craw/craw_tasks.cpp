//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file craw_tasks.cpp
//! \brief task assembly and stub task functions for CosmicRayAW.
//! Every stub returns TaskStatus::complete and does nothing yet. Fill them in.

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

namespace craw {
//----------------------------------------------------------------------------------------
//! \fn  void CosmicRayAW::AssembleCRAWTasks
//! \brief Adds craw tasks to the task lists used by the time integrator.
//! Called by MeshBlockPack::AddPhysics() directly after the CosmicRayAW constructor.
//! TODO(you): decide where feedback onto the fluid goes (after the fluid source terms
//! in stagen, as radiation's RadFluidCoupling does) and add that task here.

void CosmicRayAW::AssembleCRAWTasks(
                              std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&CosmicRayAW::InitRecv, this, none);

  // "stagen" task list
  id.copyu   = tl["stagen"]->AddTask(&CosmicRayAW::CopyCons, this, none);
  id.flux    = tl["stagen"]->AddTask(&CosmicRayAW::CalculateFluxes, this, id.copyu);
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
TaskStatus CosmicRayAW::InitRecv(Driver *, int) {
  TaskStatus tstat = pbval_u->InitRecv(nvars);
  return tstat;
}

//----------------------------------------------------------------------------------------
// stagen

//! \fn TaskStatus CosmicRayAW::CopyCons
//! \brief Copy u0 -> u1 at the start of the stage
TaskStatus CosmicRayAW::CopyCons(Driver *, int) {
  Kokkos::deep_copy(DevExeSpace(), u1, u0);
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::CalculateFluxes
//! \brief TODO(you): compute face fluxes uflx for the CR/wave variables
TaskStatus CosmicRayAW::CalculateFluxes(Driver *, int) {
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::SendFlux
//! \brief TODO(you): send fluxes to neighbouring meshblocks (SMR/AMR)
TaskStatus CosmicRayAW::SendFlux(Driver *, int) {
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::RecvFlux
//! \brief TODO(you): receive fluxes from neighbouring meshblocks (SMR/AMR)
TaskStatus CosmicRayAW::RecvFlux(Driver *, int) {
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::RKUpdate
//! \brief TODO(you): explicit RK update of u from uflx divergence
TaskStatus CosmicRayAW::RKUpdate(Driver *, int) {
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::CRAWSrcTerms
//! \brief TODO(you): source terms for the CR/wave variables (growth/damping)
TaskStatus CosmicRayAW::CRAWSrcTerms(Driver *, int) {
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
//! \brief TODO(you): set dtnew from the CR streaming/diffusion timescale.
//! Until then dtnew is set to a non-limiting value, so the CR module does not
//! restrict the mesh timestep (Mesh::NewTimeStep takes the minimum over modules).
TaskStatus CosmicRayAW::NewTimeStep(Driver *, int) {
  dtnew = std::numeric_limits<float>::max();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// after_stagen

//! \fn TaskStatus CosmicRayAW::ClearSend
//! \brief Check that all MPI sends for u have completed
TaskStatus CosmicRayAW::ClearSend(Driver *, int) {
  TaskStatus tstat = pbval_u->ClearSend();
  return tstat;
}

//! \fn TaskStatus CosmicRayAW::ClearRecv
//! \brief Check that all MPI receives for u have completed
TaskStatus CosmicRayAW::ClearRecv(Driver *, int) {
  TaskStatus tstat = pbval_u->ClearRecv();
  return tstat;
}

} // namespace craw
