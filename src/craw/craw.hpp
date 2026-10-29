#ifndef CR_AW_CR_AW_HPP_
#define CR_AW_CR_AW_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file craw.hpp
//  \brief definitions for CosmicRayAW class (cosmic rays + forward/backward Alfven
//  waves, following Thomas & Pfrommer 2019/2021, mono-energetic bin)

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class Driver;

//----------------------------------------------------------------------------------------
//! \struct CRAWTaskIDs
//  \brief container to hold TaskIDs of all craw tasks

struct CRAWTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID srctrms;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID bcs;
  TaskID prol;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace craw {

//----------------------------------------------------------------------------------------
//! \enum CRAWVariableIndex
//  \brief indices of the craw conserved variables in u0, 1D (flux along x1 only).
//  Energy densities share units with the fluid total energy density (IEN). The flux is
//  in units of velocity times energy density.
enum CRAWVariableIndex {
  ICRE = 0,   // cosmic-ray energy density
  ICRF = 1,   // cosmic-ray flux along x1 (flux along local B, 1D default)
  IAWP = 2,   // forward-propagating Alfven wave energy density (plus)
  IAWM = 3    // backward-propagating Alfven wave energy density (minus)
};

//----------------------------------------------------------------------------------------
//! \class CosmicRayAW

class CosmicRayAW {
 public:
  CosmicRayAW(MeshBlockPack *ppack, ParameterInput *pin);
  ~CosmicRayAW();

  // Number of evolved variables, set by the last CRAWVariableIndex entry
  static constexpr int nvars = IAWM + 1;

  // flags and coefficients (read from <craw> block)
  // TODO(you): add parameters here, e.g. streaming/diffusion/damping coefficients
  bool feedback;            // flag to enable/disable CR feedback on the fluid

  // conserved variables and their storage
  DvceArray5D<Real> u0;         // conserved CR/wave variables
  DvceArray5D<Real> u1;         // conserved variables at intermediate stage
  DvceArray5D<Real> coarse_u0;  // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceFaceFld5D<Real> uflx;     // spatial fluxes on zone faces
  Real dtnew;

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // container to hold names of TaskIDs
  CRAWTaskIDs id;

  // functions...
  void AssembleCRAWTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen" task list
  TaskStatus InitRecv(Driver *pdrive, int stage);
  // ...in "stagen" task list
  TaskStatus CopyCons(Driver *pdrive, int stage);
  TaskStatus CalculateFluxes(Driver *pdrive, int stage);
  TaskStatus SendFlux(Driver *pdrive, int stage);
  TaskStatus RecvFlux(Driver *pdrive, int stage);
  TaskStatus RKUpdate(Driver *pdrive, int stage);
  TaskStatus CRAWSrcTerms(Driver *pdrive, int stage);
  TaskStatus RestrictU(Driver *pdrive, int stage);
  TaskStatus SendU(Driver *pdrive, int stage);
  TaskStatus RecvU(Driver *pdrive, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *pdrive, int stage);
  TaskStatus Prolongate(Driver *pdrive, int stage);
  TaskStatus NewTimeStep(Driver *pdrive, int stage);
  // ...in "after_stagen" task list
  TaskStatus ClearSend(Driver *pdrive, int stage);
  TaskStatus ClearRecv(Driver *pdrive, int stage);

 private:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this CosmicRayAW
};

} // namespace craw
#endif // CR_AW_CR_AW_HPP_
