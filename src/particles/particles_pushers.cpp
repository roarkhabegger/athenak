//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_pushers.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "hamiltonian.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::ParticlesPush
//  \brief

TaskStatus Particles::Push(Driver *pdriver, int stage) {
  //int is = indcs.is;
  //int js = indcs.js;
  //int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &mbsize = pmy_pack->pmb->mb_size;
  //auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto dt_ = (pmy_pack->pmesh->dt);
  auto gids = pmy_pack->gids;

  switch (pusher) {
    case ParticlesPusher::drift:

      par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {
        //int m = pi(PGID,p) - gids;
        //int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
        pr(IPX,p) += 0.5*dt_*pr(IPVX,p);

        if (multi_d) {
          //int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
          pr(IPY,p) += 0.5*dt_*pr(IPVY,p);
        }

        if (three_d) {
          //int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
          pr(IPZ,p) += 0.5*dt_*pr(IPVZ,p);
        }
      });

      break;
    case ParticlesPusher::boris:
      if (not three_d) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "Boris pusher only implemented for 3D" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) { //Currently not any different than drift
        int m = prtcl_idata(PGID,p) - gids;
        //int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
        //int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
        //int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
        auto &b0_ = pmy_pack->pmhd->b0;
        auto &e0_ = pmy_pack->pmhd->efld;
        auto &u0_ = pmy_pack->pmhd->u0;
        auto &qom = charge_over_mass;

        //Push by half step using current velocities
        pr(IPX,p) += 0.5*dt_*pr(IPVX,p);
        pr(IPY,p) += 0.5*dt_*pr(IPVY,p);
        pr(IPZ,p) += 0.5*dt_*pr(IPVZ,p);

        Real uE[3], uB[3];
        Real E[3], B[3];
        Real x[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};

        InterpolateFields(x, b0_, e0_, u0_, mbsize, indcs, m, E, B);

        //Propogate velocities by electric field, half step
        uE[0] = pr(IPVX,p) + 0.5*dt_*qom*E[0];
        uE[1] = pr(IPVY,p) + 0.5*dt_*qom*E[1];
        uE[2] = pr(IPVZ,p) + 0.5*dt_*qom*E[2];

        //Propogate velocities by magnetic field, full step
        uB[0] = uE[0] + dt_*qom*(uE[1]*B[2] - uE[2]*B[1]);
        uB[1] = uE[1] + dt_*qom*(uE[2]*B[0] - uE[0]*B[2]);
        uB[2] = uE[2] + dt_*qom*(uE[0]*B[1] - uE[1]*B[0]);

        //Propogate velocities by electric field, half step
        uE[0] = uB[0] + 0.5*dt_*qom*E[0];
        uE[1] = uB[1] + 0.5*dt_*qom*E[1];
        uE[2] = uB[2] + 0.5*dt_*qom*E[2];

        //Merge velocities back into particle data
        pr(IPVX,p) = uE[0];
        pr(IPVY,p) = uE[1];
        pr(IPVZ,p) = uE[2];

        //Move particles by final half step
        pr(IPX,p) += 0.5*dt_*pr(IPVX,p);
        pr(IPY,p) += 0.5*dt_*pr(IPVY,p);
        pr(IPZ,p) += 0.5*dt_*pr(IPVZ,p);
      });
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Particle pusher not recognized" << std::endl;
      std::exit(EXIT_FAILURE);
  }

  return TaskStatus::complete;
}
} // namespace particles
