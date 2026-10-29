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
//! \brief TODO(you): post MPI receives for u
TaskStatus CosmicRayAW::InitRecv(Driver *, int) {
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// stagen

//! \fn TaskStatus CosmicRayAW::CopyCons
//! \brief TODO(you): copy u0 -> u1 at the start of the stage
TaskStatus CosmicRayAW::CopyCons(Driver *, int) {
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
//! \brief TODO(you): restrict u to coarse_u0 for SMR/AMR
TaskStatus CosmicRayAW::RestrictU(Driver *, int) {
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::SendU
//! \brief TODO(you): send u boundary data
TaskStatus CosmicRayAW::SendU(Driver *, int) {
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::RecvU
//! \brief TODO(you): receive u boundary data
TaskStatus CosmicRayAW::RecvU(Driver *, int) {
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::ApplyPhysicalBCs
//! \brief TODO(you): apply physical boundary conditions to u
TaskStatus CosmicRayAW::ApplyPhysicalBCs(Driver *, int) {
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::Prolongate
//! \brief TODO(you): prolongate u at fine/coarse boundaries with SMR/AMR
TaskStatus CosmicRayAW::Prolongate(Driver *, int) {
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
//! \brief TODO(you): check that all MPI sends for u have completed
TaskStatus CosmicRayAW::ClearSend(Driver *, int) {
  return TaskStatus::complete;
}

//! \fn TaskStatus CosmicRayAW::ClearRecv
//! \brief TODO(you): check that all MPI receives for u have completed
TaskStatus CosmicRayAW::ClearRecv(Driver *, int) {
  return TaskStatus::complete;
}

} // namespace craw
