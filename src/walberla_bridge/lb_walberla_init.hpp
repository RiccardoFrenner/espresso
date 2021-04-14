/*
 * Copyright (C) 2019-2020 The ESPResSo project
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
#ifndef LB_WALBERLA_INIT_HPP
#define LB_WALBERLA_INIT_HPP

#include "LBWalberlaBase.hpp"
#include "PE_Parameters.hpp"

#include <utils/Vector.hpp>

/** @brief Initialize Walberla's MPI manager */
void walberla_mpi_init();

LBWalberlaBase *
new_lb_walberla(double viscosity, double density,
                const Utils::Vector3i &grid_dimensions,
                const Utils::Vector3i &node_grid, double kT, unsigned int seed,
                PE_Parameters pe_params = PE_Parameters::deactivated());

#endif
