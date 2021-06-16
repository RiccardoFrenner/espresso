/*
 * Copyright (C) 2010-2021 The ESPResSo project
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "config.hpp"
#include "forces.hpp"
#include <mpi.h>
#include <stdexcept>

#if defined(VIRTUAL_SITES) && defined(LB_WALBERLA)

#include "VirtualSitesWalberlaMovingBoundary.hpp"
#include "cells.hpp"
#include "grid_based_algorithms/lb_walberla_instance.hpp"
#include "grid_based_algorithms/lb_walberla_interface.hpp"
#include "integrate.hpp"

void VirtualSitesWalberlaMovingBoundary::update() const {
  // Get the properties from the pe particle with the same id as the virtual
  // site and store its properties in the corresponding member variables of the
  // particle in espresso.
  for (auto &p : cell_structure.local_particles()) {
    if (!p.p.is_virtual)
      continue;

    auto pos = Walberla::PE_Coupling::get_particle_position(p.identity(), true);
    auto vel = Walberla::PE_Coupling::get_particle_velocity(p.identity(), true);

    // Walberla and Espresso have the same domain decomposition, therefore if
    // the pe particle is at the same position as the vs, both should be on the
    // same rank and e.g. 'pos' should not be empty. If they are not at the same
    // position, there should be at least a pe ghost particle on the same rank
    // as the vs since pe particles can not travel faster than 1 LB cell per
    // timestep and are always larger than 1 cell.
    if (!pos) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      throw std::runtime_error("There is neither a local pe particle nor a "
                               "pe ghost with uid" +
                               std::to_string(p.identity()) +
                               "on this rank (rank: " + std::to_string(rank) +
                               ")");
    }

    // LB unit conversion
    auto agrid = lb_walberla_params()->get_agrid();
    auto tau = lb_walberla_params()->get_tau();
    p.r.p = *pos * agrid;
    p.m.v = *vel * agrid / tau;

#ifdef ROTATION
    auto angular_vel = Walberla::PE_Coupling::get_particle_angular_velocity(
        p.identity(), true);
    p.m.omega = *angular_vel;
#endif

    // verlet list update check
    if ((p.r.p - p.l.p_old).norm2() > skin * skin) {
      cell_structure.set_resort_particles(Cells::RESORT_LOCAL);
    }
  }
}

// Read forces and torques from the member variables of the Espresso particle
// and add them to the walberla pe particles.
void VirtualSitesWalberlaMovingBoundary::back_transfer_forces_and_torques()
    const {
  cell_structure.ghosts_reduce_forces();

  init_forces_ghosts(cell_structure.ghost_particles());

  for (auto &p : cell_structure.local_particles()) {
    if (!p.p.is_virtual)
      continue;

    // LB unit conversion
    auto agrid = lb_walberla_params()->get_agrid();
    auto tau = lb_walberla_params()->get_tau();
    auto lb_force = p.f.f * tau * tau / agrid;

    Walberla::PE_Coupling::add_particle_force(p.identity(), lb_force);

#ifdef ROTATION
    // LB unit conversion
    auto lb_torque = p.f.torque * tau * tau / (agrid * agrid);

    Walberla::PE_Coupling::add_particle_torque(p.identity(), lb_torque);
#endif
  }
}
#endif