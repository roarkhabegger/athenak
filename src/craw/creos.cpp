//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file creos.cpp
//! \brief constructor for CREOS, reads the CR EOS parameters and floors from <craw>

#include <cfloat>   // FLT_MIN

#include "athena.hpp"
#include "parameter_input.hpp"
#include "craw/creos.hpp"

//----------------------------------------------------------------------------------------
//! \fn CREOS::CREOS
//! \brief Reads gamma_cr, gamma_a and the CR floors. Floors default to FLT_MIN, the same
//! placeholder the fluid EOS uses for dfloor/pfloor/tfloor/sfloor.

CREOS::CREOS(ParameterInput *pin) {
  eos_data.gamma_cr = 4.0/3.0;
  eos_data.gamma_a  = 3.0/2.0;
  eos_data.efloor = pin->GetOrAddReal("craw","efloor",(FLT_MIN));
  eos_data.wfloor = pin->GetOrAddReal("craw","wfloor",(FLT_MIN));
}
