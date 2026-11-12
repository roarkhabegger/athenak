#ifndef CRAW_CREOS_HPP_
#define CRAW_CREOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file creos.hpp
//! \brief Equation of state for the cosmic-ray and Alfven-wave variables, mirroring
//! EquationOfState for the fluid. CREOS_Data holds what kernels capture by value;
//! CREOS owns it and is constructed from the <craw> block. The CR floors live here and
//! nowhere else. The CR variable index enum lives here so that craw.hpp can include this
//! header without a cycle.

#include "athena.hpp"
#include "parameter_input.hpp"

namespace craw {

//----------------------------------------------------------------------------------------
//! \enum CRAWVariableIndex
//  \brief indices of the craw conserved variables in u0, 1D (flux along x1 only).
//  These index craw's own u0 array, so they may share numbers with fluid indices (e.g.
//  ICRE == IDN == 0) without collision: the array they index is different.
enum CRAWVariableIndex {
  ICRE = 0,   // cosmic-ray energy density
  ICRF = 1,   // cosmic-ray flux along x1 (flux along local B, 1D default)
  IAWP = 2,   // forward-propagating Alfven wave energy density (plus)
  IAWM = 3    // backward-propagating Alfven wave energy density (minus)
};

} // namespace craw

//----------------------------------------------------------------------------------------
//! \struct CREOS_Data
//! \brief container for CR EOS variables and functions needed inside kernels, captured by
//! value like EOS_Data.

struct CREOS_Data {
  Real gamma_cr;   // adiabatic index of CRs (4/3)
  Real gamma_a;    // adiabatic index of Alfven waves (3/2)
  Real efloor;     // floor on CR energy density ICRE
  Real wfloor;     // floor on wave energy densities IAWP, IAWM

  // CR PRESSURE: P_cr = (gamma_cr - 1) e_cr
  KOKKOS_INLINE_FUNCTION
  Real CRPressure(const Real ecr) const {
    return (gamma_cr - 1.0)*ecr;
  }

  // ALFVEN WAVE PRESSURE: P_a = (gamma_a - 1) e_a, for one wave family
  KOKKOS_INLINE_FUNCTION
  Real WavePressure(const Real ea) const {
    return (gamma_a - 1.0)*ea;
  }

  // FLOOR for slot n of CR data. ICRE and the wave energies are positive-definite and are
  // floored. ICRF (CR flux) is signed and is not floored.
  KOKKOS_INLINE_FUNCTION
  Real FloorSlot(const int n, const Real val) const {
    if (n == craw::ICRE) return fmax(val, efloor);
    if (n == craw::IAWP || n == craw::IAWM) return fmax(val, wfloor);
    return val;
  }
};

//----------------------------------------------------------------------------------------
//! \class CREOS
//! \brief owns CREOS_Data, read from the <craw> block.

class CREOS {
 public:
  explicit CREOS(ParameterInput *pin);
  ~CREOS() = default;

  CREOS_Data eos_data;
};

#endif // CRAW_CREOS_HPP_
