#pragma once

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
#include "config.hpp"
#include "utils/constants.hpp"
#include <algorithm>
#include <boost/test/tools/detail/print_helper.hpp>
#include <cstdint>

#ifdef LB_WALBERLA

#include <LBWalberlaBase.hpp>
#include <LBWalberlaD3Q19FluctuatingMRT.hpp>
#include <LBWalberlaD3Q19MRT.hpp>
#include <PE_Parameters.hpp>
#include <lb_walberla_init.hpp>
#include <walberla_utils.hpp>

#include <utils/Vector.hpp>

#include <functional>
#include <memory>
#include <vector>

class LBTestParameters {
public:
  int seed;
  double kT;
  double viscosity;
  double density;
  Utils::Vector3d box_dimensions;
  Utils::Vector3i grid_dimensions;

  LBTestParameters(int seed_, double kT_, double viscosity_, double density_,
                   Utils::Vector3d box_dimensions_,
                   Utils::Vector3i grid_dimensions_)
      : seed(seed_), kT(kT_), viscosity(viscosity_), density(density_),
        box_dimensions(box_dimensions_), grid_dimensions(grid_dimensions_) {}

  Utils::Vector3i n_blocks;
  double particle_density;
  double particle_radius;
  Utils::Vector3d particle_initial_position;
  Utils::Vector3d particle_initial_velocity;
  std::vector<std::pair<Utils::Vector3d, std::string>> external_particle_forces;
  bool force_avg = true;

  double get_particle_volume() {
    return 4. / 3. * Utils::pi() * std::pow(particle_radius, 3);
  }
  double get_particle_mass() {
    return get_particle_volume() * particle_density;
  }
  Utils::Vector3d get_particle_initial_momentum() {
    return get_particle_mass() * particle_initial_velocity;
  }

  LBTestParameters(Utils::Vector3i grid_dimensions_ = Utils::Vector3i{100, 100,
                                                                      160})
      : seed(0), kT(0), viscosity(5e-3), density(1.),
        box_dimensions({double(grid_dimensions_[0]),
                        double(grid_dimensions_[1]),
                        double(grid_dimensions_[2])}),
        grid_dimensions(grid_dimensions_), particle_density(1.1),
        particle_initial_velocity({0, 0, 0}), external_particle_forces() {
    auto min_length =
        *std::min_element(grid_dimensions_.begin(), grid_dimensions_.end());
    particle_radius = std::max(min_length * .1, 2.);
    particle_initial_position =
        Utils::Vector3d{.5 * grid_dimensions_[0], .5 * grid_dimensions_[1],
                        2 * particle_radius};
  }
  LBTestParameters(Utils::Vector3i n_blocks_,
                   Utils::Vector3i grid_dimensions_ = Utils::Vector3i{100, 100,
                                                                      160},
                   bool force_avg_ = true)
      : seed(0), kT(0), viscosity(5e-3), density(1.),
        box_dimensions({double(grid_dimensions_[0]),
                        double(grid_dimensions_[1]),
                        double(grid_dimensions_[2])}),
        grid_dimensions(grid_dimensions_), n_blocks(n_blocks_),
        particle_density(1.1), particle_initial_velocity({0, 0, 0}),
        external_particle_forces(), force_avg(force_avg_) {
    int min_length =
        *(std::min_element(grid_dimensions_.begin(), grid_dimensions_.end()));
    particle_radius = std::max(.1 * min_length, 2.);
    particle_initial_position =
        Utils::Vector3d{.5 * grid_dimensions_[0], .5 * grid_dimensions_[1],
                        2 * particle_radius};
  }
};

using LbGeneratorVector =
    std::vector<std::function<std::shared_ptr<LBWalberlaBase>(
        Utils::Vector3i, LBTestParameters)>>;

// Add all LBs with kT=0 to be tested, here
LbGeneratorVector unthermalized_lbs() {
  LbGeneratorVector lbs;
  // Unthermalized D3Q19 MRT
  lbs.push_back([](const Utils::Vector3i mpi_shape,
                   const LBTestParameters &params) {
    return std::make_shared<walberla::LBWalberlaD3Q19MRT>(
        params.viscosity, params.density, params.grid_dimensions, mpi_shape, 1);
  });

  // Thermalized D3Q19 MRT with kT set to 0
  lbs.push_back([](Utils::Vector3i mpi_shape, const LBTestParameters &params) {
    return std::make_shared<walberla::LBWalberlaD3Q19FluctuatingMRT>(
        params.viscosity, params.density, params.grid_dimensions, mpi_shape, 1,
        0.0, params.seed);
  });
  return lbs;
}

// Add all LBs with thermalization to be tested, here
LbGeneratorVector thermalized_lbs() {
  LbGeneratorVector lbs;

  // Thermalized D3Q19 MRT with kT set to 0
  lbs.push_back(
      [](const Utils::Vector3i mpi_shape, const LBTestParameters &params) {
        return std::make_shared<walberla::LBWalberlaD3Q19FluctuatingMRT>(
            params.viscosity, params.density, params.grid_dimensions, mpi_shape,
            1, params.kT, params.seed);
      });
  return lbs;
}

LbGeneratorVector all_lbs() {
  auto lbs = unthermalized_lbs();
  auto thermalized = thermalized_lbs();
  lbs.insert(lbs.end(), thermalized.begin(), thermalized.end());
  return lbs;
}

// Disable printing of type which does not support it
BOOST_TEST_DONT_PRINT_LOG_VALUE(LbGeneratorVector::value_type)

std::vector<Utils::Vector3i>
all_nodes_incl_ghosts(const Utils::Vector3i &grid_dimensions,
                      int n_ghost_layers) {
  std::vector<Utils::Vector3i> res;
  for (int x = -n_ghost_layers; x < grid_dimensions[0] + n_ghost_layers; x++) {
    for (int y = -n_ghost_layers; y < grid_dimensions[1] + n_ghost_layers;
         y++) {
      for (int z = -n_ghost_layers; z < grid_dimensions[2] + n_ghost_layers;
           z++) {
        res.push_back(Utils::Vector3i{x, y, z});
      }
    }
  }
  return res;
}

std::vector<Utils::Vector3i> local_nodes_incl_ghosts(
    std::pair<Utils::Vector3d, Utils::Vector3d> local_domain,
    int n_ghost_layers) {
  std::vector<Utils::Vector3i> res;
  auto const left = local_domain.first;
  auto const right = local_domain.second;

  for (int x = static_cast<int>(left[0]) - n_ghost_layers;
       x < static_cast<int>(right[0]) + n_ghost_layers; x++) {
    for (int y = static_cast<int>(left[1]) - n_ghost_layers;
         y < static_cast<int>(right[1]) + n_ghost_layers; y++) {
      for (int z = static_cast<int>(left[2]) - n_ghost_layers;
           z < static_cast<int>(right[2]) + n_ghost_layers; z++) {
        res.push_back(Utils::Vector3i{x, y, z});
      }
    }
  }
  return res;
}

std::vector<Utils::Vector3i> corner_nodes(Utils::Vector3i n) {
  std::vector<Utils::Vector3i> res;
  for (int i : {0, n[0] - 1})
    for (int j : {0, n[1] - 1})
      for (int k : {0, n[2] - 1})
        res.emplace_back(Utils::Vector3i{i, j, k});
  return res;
}

#endif
