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
#ifndef VIRTUAL_SITES_VIRTUAL_SITES_WALBERLA_MOVING_BOUNDARY_HPP
#define VIRTUAL_SITES_VIRTUAL_SITES_WALBERLA_MOVING_BOUNDARY_HPP

#include "config.hpp"
#ifdef VIRTUAL_SITES
#include "VirtualSites.hpp"

#if defined(VIRTUAL_SITES) && defined(LB_WALBERLA)
/** @brief Virtual sites acting as "shadows" for the particles in Walberla's
 *         physics engine (PE).
 */
class VirtualSitesWalberlaMovingBoundary : public VirtualSites {
  void update() const override;
  void back_transfer_forces_and_torques() const override;
};

#endif
#endif
#endif
