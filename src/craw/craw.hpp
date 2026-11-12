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
#include "craw/creos.hpp"   // CREOS, CREOS_Data, and the CRAWVariableIndex enum

// forward declarations
class Driver;

//----------------------------------------------------------------------------------------
//! \struct CRAWTaskIDs
//  \brief container to hold TaskIDs of all craw tasks

struct CRAWTaskIDs {
  TaskID irecv_sub;   // before_craw_sub: post receives for u
  TaskID irecv;       // before_stagen: post receives for u, every stage
  TaskID copyu;  // CopyCons, stagen: u0 -> u1 at stage 1
  TaskID sendu, recvu, prol, bcs;  // u0 ghost exchange in stagen, every stage
  TaskID rkupdt;                   // RKUpdate, stagen: adiabatic CR update, after flux
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID sub_rkupdt;  // Sub_RKUpdate, craw_sub: anisotropic subcycle update
  TaskID srctrms;
  TaskID restu;
  TaskID sendu_sub;
  TaskID recvu_sub;
  TaskID bcs_sub;
  TaskID prol_sub;
  TaskID newdt;
  TaskID csend_sub;   // after_craw_sub: finish sends of u
  TaskID crecv_sub;   // after_craw_sub: finish receives of u
  TaskID csend;       // after_stagen: finish sends of u, every stage
  TaskID crecv;       // after_stagen: finish receives of u, every stage
};

namespace craw {

//----------------------------------------------------------------------------------------
//! \class CosmicRayAW

class CosmicRayAW {
 public:
  CosmicRayAW(MeshBlockPack *ppack, ParameterInput *pin);
  ~CosmicRayAW();

  // Number of evolved variables, set by the last CRAWVariableIndex entry
  static constexpr int nvars = IAWM + 1;

  // adiabatic index of the cosmic-ray gas (relativistic CRs: 4/3)
  static constexpr Real gamma_cr = 4.0/3.0;

  // flags and coefficients (read from <craw> block)
  // TODO(roark): add parameters here, e.g. streaming/diffusion/damping coefficients
  Real v_max;               // reduced speed of light for the CR f_c equation
  Real v_A;                 // Alfven speed; read only when no fluid is present
  Real lorentz;             // Lorentz factor of the mono-energetic CR bin (fixed)
  Real alpha_landau;        // nonlinear Landau damping coefficient (alpha)
  Real atol;                // absolute tolerance for the source-term error estimate
  Real rtol;                // relative tolerance for the source-term error estimate
  CREOS *pcreos;            // CR equation of state: pressures and floors (owns the floors)
  Real m_cr;               // CR particle mass in code units (fixed, default proton mass)
  Real B_code;              // standalone field (code units) used to set chi
  Real chi_cr;              // chi = (9 pi/8) lorentz m c^3 B, B = B_code (standalone only)
  // B-independent coefficients and code-unit scales, shared by the standalone and MHD paths:
  //   chi = chi_K / B_gauss,   alpha = (alpha_K / B_gauss) sqrt(P/rho) v_unit E_unit T_unit
  Real chi_K;               // gyroresonant coefficient, cgs, chi = chi_K / B_gauss
  Real alpha_K;             // Landau coefficient, cgs, see above
  Real E_unit, v_unit, T_unit;  // code scales: energy density (erg/cm^3), velocity (cm/s), time (s)
  bool has_fluid;           // true if hydro or MHD is present in the pack

  // conserved variables and their storage
  DvceArray5D<Real> u0;         // conserved CR/wave variables
  DvceArray5D<Real> u1;         // conserved variables at intermediate stage
  int N_cr;                     // CR substeps per fluid timestep, used in NewTimeStep (ncr_sub)
  DvceArray5D<Real> coarse_u0;  // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceFaceFld5D<Real> uflx;     // spatial fluxes on zone faces (exchanged)
  DvceFaceFld5D<Real> uflx_aniso;  // anisotropic transport fluxes on zone faces
                                   // (x1: ICRF holds the Riemann star value f_c*, not S f_c*)

  // adiabatic CR step (2010 Sec. 2.1), used with MHD. Reconstructed L/R states of u0 for
  // each face direction, filled by ReconDispatch with the MHD reconstruction method.
  DvceArray5D<Real> wl_cr, wr_cr;       // (nmb, nvars, n3, n2, n1), reconstructed u0 (x1)
  // Face outputs of the CR-aware HLLD, for all three face directions as for uflx_aniso:
  // contact velocity u* on each face, and the advective CR fluxes on each face
  DvceFaceFld4D<Real> uface_cr;         // contact velocity u* on faces
  DvceFaceFld5D<Real> crflx_adv;        // advective CR fluxes on faces, nvars components
  Real dtnew;

  // wall-clock time spent in the craw subcycle task lists, summed over the run
  Kokkos::Timer subcycle_timer;
  double subcycle_seconds;
  int nsubcycles;

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // container to hold names of TaskIDs
  CRAWTaskIDs id;

  // functions...
  void AssembleCRAWTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskID CRAWFluidFluxDependency(TaskID &own_copyu);
  // ...in "before_stagen" task list
  TaskStatus InitRecv(Driver *pdrive, int stage);
  // ...in "stagen" task list
  TaskStatus CopyCons(Driver *pdrive, int stage);
  TaskStatus CalculateAnisoFluxes(Driver *pdrive, int stage);
  TaskStatus SendFlux(Driver *pdrive, int stage);
  TaskStatus RecvFlux(Driver *pdrive, int stage);
  TaskStatus Sub_RKUpdate(Driver *pdrive, int stage);  // anisotropic subcycle update
  TaskStatus RKUpdate(Driver *pdrive, int stage);      // adiabatic CR update, stagen
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
