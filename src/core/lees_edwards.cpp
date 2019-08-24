/*
  Copyright (C) 2018 The ESPResSo project

  This file is part of ESPResSo.

  ESPResSo is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ESPResSo is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
/** \file lees_edwards.cpp
 */

#include "lees_edwards.hpp"
#include "integrate.hpp"
#include "grid.hpp"
#include <cmath>

namespace LeesEdwards {
#ifdef LEES_EDWARDS
void local_image_reset(const ParticleRange &particles) {
  for (auto &p : particles) {
    p.l.i = Utils::Vector3i::broadcast(0.);
    p.p.lees_edwards_offset = 0;
  }
}

double pos_offset_at_verlet_rebuild;

void on_verlet_rebuild() {
 pos_offset_at_verlet_rebuild = get_pos_offset(sim_time, box_geo.lees_edwards_protocol);
}

}
#endif
