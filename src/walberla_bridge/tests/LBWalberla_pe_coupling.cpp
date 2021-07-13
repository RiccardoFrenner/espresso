/*
 * Copyright (C) 2020-2021 The ESPResSo project
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

#include "stencil/D3Q27.h"
#include "utils/constants.hpp"
#include "utils/quaternion.hpp"
#include <algorithm>
#include <boost/qvm/vec_operations.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mpi.h>
#define BOOST_TEST_MODULE Walberla pe setters and getters test
#define BOOST_TEST_DYN_LINK
#include "config.hpp"

#ifdef LB_WALBERLA

#define BOOST_TEST_NO_MAIN

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <memory>
#include <vector>

#include "PE_Parameters.hpp"
#include "tests_common.hpp"
#include "utils/Vector.hpp"
#include <lb_walberla_init.hpp>

using Utils::Quaternion;
using Utils::Vector3d;
using Utils::Vector3i;

namespace bdata = boost::unit_test::data;

LBTestParameters params; // populated in main()
Vector3i mpi_shape;      // populated in main()
int time_steps;          // populated in main()

BOOST_AUTO_TEST_CASE(particle_setters_getters) {
  auto lb = std::make_shared<walberla::LBWalberlaD3Q19MRT>(
      params.viscosity, params.density, params.grid_dimensions, mpi_shape, 1,
      PE_Parameters());

  // Create particle
  std::uint64_t uid = 12;
  double radius = 0.1;
  Vector3d global_pos{0, 0, 0};
  lb->add_particle(uid, global_pos, radius);
  lb->finish_particle_adding();

  // Set attributes
  Quaternion<double> orientation{1., 0., 0., 0.};
  Vector3d linear_vel{1.0, 0.2, 0.1};
  Vector3d angular_vel{1.1, 0.1, 0.3};
  Vector3d force{0.1, 0.5, 0.22};
  Vector3d torque{0.5, 0.1, 0.324};
  lb->set_particle_orientation(uid, orientation);
  lb->set_particle_velocity(uid, linear_vel);
  lb->set_particle_angular_velocity(uid, angular_vel);
  lb->set_particle_force(uid, force);
  lb->set_particle_torque(uid, torque);

  // Check that particle exists exactly on one rank
  int exists = lb->is_particle_on_this_process(uid) ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &exists, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  BOOST_CHECK(exists == 1);

  if (lb->is_particle_on_this_process(uid)) {
    BOOST_CHECK(*(lb->get_particle_position(uid)) == global_pos);
    BOOST_CHECK(*(lb->get_particle_orientation(uid)) == orientation);
    BOOST_CHECK(*(lb->get_particle_velocity(uid)) == linear_vel);
    BOOST_CHECK(*(lb->get_particle_angular_velocity(uid)) == angular_vel);
    BOOST_CHECK(*(lb->get_particle_force(uid)) == force);
    BOOST_CHECK(*(lb->get_particle_torque(uid)) == torque);
  } else {
    // Check that access to particle attributes is not possible
    BOOST_CHECK(!lb->get_particle_position(uid));
    BOOST_CHECK(!lb->get_particle_orientation(uid));
    BOOST_CHECK(!lb->get_particle_velocity(uid));
    BOOST_CHECK(!lb->get_particle_angular_velocity(uid));
    BOOST_CHECK(!lb->get_particle_force(uid));
    BOOST_CHECK(!lb->get_particle_torque(uid));
  }
}

BOOST_AUTO_TEST_CASE(remove_particle) {
  auto lb = std::make_shared<walberla::LBWalberlaD3Q19MRT>(
      params.viscosity, params.density, params.grid_dimensions, mpi_shape, 1,
      PE_Parameters());

  // Create particle
  std::uint64_t uid = 12;
  double radius = 0.1;
  Vector3d global_pos{0, 0, 0};
  lb->add_particle(uid, global_pos, radius);
  lb->finish_particle_adding();

  lb->remove_particle(uid);

  // Check that particle doesn't exist on any rank
  BOOST_CHECK(lb->is_particle_on_this_process(uid, true) == false);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int n_nodes;

  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
  MPI_Dims_create(n_nodes, 3, mpi_shape.data());

  Vector3i grid_dimension{24, 24, 24};
  Vector3i block_dimension{1, 1, 1};
  Vector3d particle_pos{12, 12, 12};
  Vector3d particle_vel{0, 0, 0};
  double particle_radius{3};
  bool force_avg = true;
  time_steps = 10;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--") == 0) { // Needed because of boost-test
      continue;
    }
    if (std::strcmp(argv[i], "--resolution") == 0) {
      int r = std::atoi(argv[++i]);
      grid_dimension = {r, r, r};
      particle_pos = .5 * grid_dimension;
      continue;
    }
    if (std::strcmp(argv[i], "--grid_dim") == 0) {
      grid_dimension = {std::atoi(argv[++i]), std::atoi(argv[++i]),
                        std::atoi(argv[++i])};
      particle_pos = .5 * grid_dimension;
      continue;
    }
    if (std::strcmp(argv[i], "--xpos") == 0) {
      particle_pos[0] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--ypos") == 0) {
      particle_pos[1] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--zpos") == 0) {
      particle_pos[2] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--xvel") == 0) {
      particle_vel[0] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--yvel") == 0) {
      particle_vel[1] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--zvel") == 0) {
      particle_vel[2] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--with_force_avg") == 0) {
      force_avg = true;
      continue;
    }
    if (std::strcmp(argv[i], "--radius") == 0) {
      particle_radius = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--timesteps") == 0) {
      time_steps = std::atoi(argv[++i]);
      continue;
    }
    std::cout << "Unrecognized command line argument found: " << argv[i]
              << std::endl;
    return 1;
  }
  params = LBTestParameters(block_dimension, grid_dimension, force_avg);
  params.particle_radius = particle_radius;
  params.particle_initial_position = particle_pos;
  params.particle_initial_velocity = particle_vel;

  walberla_mpi_init();
  auto res = boost::unit_test::unit_test_main(init_unit_test, argc, argv);
  MPI_Finalize();
  return res;
}

#else  // ifdef LB_WALBERLA
int main(int argc, char **argv) {}
#endif // ifdef LB_WALBERLA
