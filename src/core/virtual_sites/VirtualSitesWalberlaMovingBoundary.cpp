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
#include <mpi.h>
#include <stdexcept>

#if defined(VIRTUAL_SITES) && defined(LB_WALBERLA)

#include "VirtualSitesWalberlaMovingBoundary.hpp"
#include "cells.hpp"
#include "grid_based_algorithms/lb_walberla_interface.hpp"

void VirtualSitesWalberlaMovingBoundary::update() const {
  // Get the properties from the pe particle with the same id as the virtual
  // site and store its properties in the corresponding member variables of the
  // particle in espresso.
  for (auto &p : cell_structure.local_particles()) {
    if (!p.p.is_virtual)
      continue;

    auto pos = Walberla::PE_Coupling::get_particle_position(p.identity());
    auto vel = Walberla::PE_Coupling::get_particle_velocity(p.identity());

    // Walberla and Espresso have the same domain decomposition, therefore if
    // the pe particle is at the same position as the vs, both should be on the
    // same rank and pos e.g. should contain the position.
    // If they are not at the same position, there should be at least a pe ghost
    // particle on the same rank as the vs since pe particles can not travel
    // faster than 1 LB cell per timestep and are always larger than 1 cell.

    // todo: Problem is I don't update the particle map in LBWalberlaImpl when
    // the particle goes from one block to another. Therefore this execption
    // is expected to occur once the particle changes blocks.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Rank: " << rank << std::endl;
    if (!pos) {
      std::cout << "hghghghghghgh" << std::endl;
      throw std::runtime_error("Pe particle and vs are not on the same rank!");
    }

    p.r.p = *pos;
    p.m.v = *vel;

#ifdef ROTATION
    auto angular_vel =
        Walberla::PE_Coupling::get_particle_angular_velocity(p.identity());
    p.m.omega = *angular_vel;
#endif
  }
}
void VirtualSitesWalberlaMovingBoundary::back_transfer_forces_and_torques()
    const {

  // Communication needs to happen here since walberla pe particles only exist
  // on one rank and have no ghosts. Therefore the pe particle corresponding to
  // a ghost particle on this rank cannot be accessed here to transfer forces.
  cell_structure.ghosts_reduce_forces();

  // Read forces and torques from the member variables of the Espresso particle
  // and add them to the walberla pe particles.
  for (auto &p : cell_structure.local_particles()) {
    if (!p.p.is_virtual)
      continue;

    auto f = p.f.f;
    Walberla::PE_Coupling::add_particle_force(p.identity(), f);
#ifdef ROTATION
    auto tau = p.f.torque;
    Walberla::PE_Coupling::add_particle_torque(p.identity(), tau);
#endif
  }
}
#endif