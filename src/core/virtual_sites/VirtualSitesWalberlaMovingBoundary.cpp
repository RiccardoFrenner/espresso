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

#if defined(VIRTUAL_SITES) && defined(LB_WALBERLA)

#include "VirtualSitesWalberlaMovingBoundary.hpp"
#include "cells.hpp"
#include "grid_based_algorithms/lb_interface.hpp"

void VirtualSitesWalberlaMovingBoundary::update() const {
  // Get the properties from the pe particle with the same id as the virtual
  // site and store its properties in the corresponding member variables of the
  // particle in espresso.
  for (auto &p : cell_structure.local_particles()) {
    if (!p.p.is_virtual)
      continue;

    auto pos = pe_get_particle_position(p.identity());
    auto vel = pe_get_particle_velocity(p.identity());

    p.r.p = pos;
    p.m.v = vel;

#ifdef ROTATION
    auto angular_vel = pe_get_particle_angular_velocity(p.identity());
    p.m.omega = angular_vel;
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
    pe_add_particle_force(p.identity(), f);
#ifdef ROTATION
    auto tau = p.f.torque;
    pe_add_particle_torque(p.identity(), tau);
#endif
  }
}
#endif