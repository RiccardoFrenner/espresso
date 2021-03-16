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

#ifdef VIRTUAL_SITES_INERTIALESS_TRACERS
#include "VirtualSitesWalberlaMovingBoundary.hpp"
#include "cells.hpp"
#include "grid_based_algorithms/lb_interface.hpp"

void VirtualSitesWalberlaMovingBoundary::update() const {
  // Get the properties from the PE particle with the same id as the virtual
  // site and store its properties (such as pos, vel) in the corresponding
  // member variables of the Particle in Espresso
  for (auto &p : cell_structure.local_particles()) {
    if (!p.p.is_virtual)
      continue;

    // TODO: check id
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
  // Read forces and torques from the member variables of the Espresso particle
  // and call the add force/torque methods from the Pe particle registry
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

  // TODO: Ghost particles add their forces to their corresponding real
  // particles. But does that happen before I get the real particle's properties
  // and apply them to the pe particles? If not I have to iterate through the
  // ghost particles here as well and apply their forces to the pe particles.
  // But those wouldn't be on this rank ...
}
#endif
