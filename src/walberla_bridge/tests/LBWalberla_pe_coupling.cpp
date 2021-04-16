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

#include <algorithm>
#include <boost/test/tools/old/interface.hpp>
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

using Utils::Vector3d;
using Utils::Vector3i;

namespace bdata = boost::unit_test::data;

LBTestParameters params; // populated in main()
Vector3i mpi_shape;      // populated in main()

// BOOST_AUTO_TEST_CASE(add_particle_inside_domain) {
//   auto lb = std::make_shared<LBWalberlaD3Q19MRT>(
//       viscosity, density, grid_dimensions, mpi_shape, 1, PE_Parameters());

//   // Create particle
//   std::uint64_t uid = 12;
//   double radius = 0.1;
//   Vector3d global_pos{0, 0, 0};
//   Vector3d linear_vel{1.0, 0.2, 0.1};
//   lb->add_particle(uid, global_pos, radius, linear_vel);
//   lb->finish_particle_adding();

//   // Add force/torque
//   Vector3d force{0.1, 0.5, 0.22};
//   Vector3d torque{0.5, 0.1, 0.324};
//   lb->set_particle_force(uid, force);
//   lb->set_particle_torque(uid, torque);

//   // Check that particle exists exactly on one rank
//   int exists = lb->is_particle_on_this_process(uid) ? 1 : 0;
//   MPI_Allreduce(MPI_IN_PLACE, &exists, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//   BOOST_CHECK(exists == 1);

//   if (lb->is_particle_on_this_process(uid)) {
//     BOOST_CHECK(*(lb->get_particle_position(uid)) == global_pos);
//     BOOST_CHECK(*(lb->get_particle_velocity(uid)) == linear_vel);
//     BOOST_CHECK(*(lb->get_particle_force(uid)) == force);
//     BOOST_CHECK(*(lb->get_particle_torque(uid)) == torque);
//     // TODO: setter for angular vel and orientation
//     // BOOST_CHECK(*(lb->get_particle_angular_velocity()) == angularVel);
//     // BOOST_CHECK(*(lb->get_particle_orientation()) == orientation);
//   } else {
//     // Check that access to particle attributes is not possible
//     BOOST_CHECK(!lb->get_particle_position(uid));
//     BOOST_CHECK(!lb->get_particle_velocity(uid));
//   }
// }

// BOOST_AUTO_TEST_CASE(remove_particle) {
//   auto lb = std::make_shared<LBWalberlaD3Q19MRT>(
//       viscosity, density, grid_dimensions, mpi_shape, 1, PE_Parameters());

//   // Create particle
//   std::uint64_t uid = 12;
//   double radius = 0.1;
//   Vector3d global_pos{0, 0, 0};
//   Vector3d linear_vel{1.0, 0.2, 0.1};
//   lb->add_particle(uid, global_pos, radius, linear_vel);
//   lb->finish_particle_adding();

//   // Add force/torque
//   Vector3d force{0.1, 0.5, 0.22};
//   Vector3d torque{0.5, 0.1, 0.324};
//   lb->set_particle_force(uid, force);
//   lb->set_particle_torque(uid, torque);

//   lb->remove_particle(uid);

//   // Check that particle doesn't exist on any rank
//   BOOST_CHECK(lb->is_particle_on_this_process(uid) == false);
// }

// BOOST_DATA_TEST_CASE(no_external_forces, bdata::make(pe_enabled_lbs()),
//                      lb_generator) {
//   auto lb = lb_generator(mpi_shape, params);

//   constexpr uint64_t uid = 0;
//   constexpr uint64_t steps = 100;

//   // Check force == 0 and no movement
//   if (lb->is_particle_on_this_process(uid)) {
//     auto initial_position = *(lb->get_particle_position(uid));
//     for (uint64_t i = 0; i < steps; ++i) {
//       lb->integrate();
//       auto pos = *(lb->get_particle_position(uid));
//       auto f = *(lb->get_particle_force(uid));
//       BOOST_CHECK_SMALL(f.norm(), 1e-10);
//       BOOST_CHECK_SMALL((initial_position - pos).norm(), 1e-10);
//     }
//   }
// }

void write_data(uint64_t timestep, std::vector<Vector3d> vectors,
                std::string const &filename) {
  std::ofstream file;
  file.precision(5);
  file.setf(std::ofstream::fixed);
  file.open(filename.c_str(), std::ofstream::app);

  file << "|" << std::setw(4) << timestep << " |";
  for (auto const &vec : vectors) {
    for (uint64_t i = 0; i < 3; ++i) {
      file << std::setw(8) << vec[i] << " ";
    }
    file << "(" << std::setw(8) << vec.norm() << ") |";
  }
  file << std::endl;
  file.close();
}

BOOST_DATA_TEST_CASE(momentum_conservation, bdata::make(pe_enabled_lbs()),
                     lb_generator) {
  params.particle_initial_velocity = Vector3d{.1, 0, 0};
  auto lb = lb_generator(mpi_shape, params);

  // check fluid has no momentum
  Vector3d fluid_mom = lb->get_momentum();
  printf("fluid_mom: %f %f %f, norm: %f\n", fluid_mom[0], fluid_mom[1],
         fluid_mom[2], fluid_mom.norm());
  MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  BOOST_CHECK_SMALL(fluid_mom.norm(), 1e-10);

  constexpr uint64_t uid = 0;
  constexpr uint64_t steps = 100;

  std::ofstream file;
  std::string filename{"data/momentum_conservation.txt"};
  file.open(filename.c_str());
  file << "|  #  |"
          "            fluid momentum            |"
          "           particle velocity          |"
          "           particle position          |"
          "            particle force            |"
       << std::endl;
  file.close();
  Vector3d particle_mom{};
  Vector3d particle_pos{};
  Vector3d particle_force{};
  Vector3d particle_ang_vel{};
  for (uint64_t i = 0; i < steps; ++i) {
    fluid_mom = lb->get_momentum();
    if (lb->is_particle_on_this_process(uid)) {
      particle_mom =
          *(lb->get_particle_velocity(uid)) * params.get_particle_mass();
      particle_force = *(lb->get_particle_force(uid));
      particle_pos = *(lb->get_particle_position(uid));
      particle_ang_vel = *(lb->get_particle_angular_velocity(uid));
    }
    write_data(i,
               {fluid_mom, particle_mom / params.get_particle_mass(),
                particle_pos, particle_ang_vel},
               filename);
    if (std::any_of(fluid_mom.begin(), fluid_mom.end(),
                    [](double a) { return std::isnan(a); }))
      break;
    lb->integrate();
  }
  MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, particle_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  BOOST_CHECK_SMALL((particle_mom - fluid_mom).norm(), 1e-10);
}

// BOOST_AUTO_TEST_CASE(getting_forces) {
//   printf("MPI-Shape: %d %d %d\n", mpi_shape[0], mpi_shape[1],
//   mpi_shape[2]); std::vector<std::pair<Vector3d, std::string>> ext_forces{
//       {Vector3d{0, 0, -(1.166666 - 1.0) * 6.565115e-5 * 1.767e3},
//        "Test force"}};
//   PE_Parameters pe_params(ext_forces);
//   BOOST_CHECK(pe_params.is_activated());
//   // auto lb = std::make_shared<LBWalberlaD3Q19MRT>(
//   //     4.942463e-3, 1.0, Vector3i{100, 100, 160}, mpi_shape, 1,
//   pe_params); auto lb = std::make_shared<LBWalberlaD3Q19MRT>(
//       4.942463e-3, 1.0, Vector3i{2, 2, 1}, Vector3i{50, 50, 160},
//       mpi_shape, 1, pe_params);

//   // Create particle
//   std::uint64_t uid = 12;
//   double radius = 7.5;
//   Vector3d global_pos{50, 50, 20};
//   Vector3d linear_vel{.0, .0, .0};
//   lb->create_particle_material("myMat", 1.166666, 0.5, 0.1, 0.1, 0.24, 200,
//   200,
//                                0, 0);
//   lb->add_particle(uid, global_pos, radius, linear_vel, "myMat");
//   lb->finish_particle_adding();

//   // Check force == 0 before integration
//   auto f = lb->get_particle_force(uid);
//   if (f)
//     printf("Force: (%f, %f, %f)\n", (*f)[0], (*f)[1], (*f)[2]);
//   if (lb->is_particle_on_this_process(uid)) {
//     auto n = (*f).norm();
//     BOOST_CHECK_SMALL(n, 1e-10);
//   } else {
//     BOOST_CHECK(!f);
//   }

//   // Check force != 0 after integration
//   lb->integrate();
//   // lb->integrate();
//   f = lb->get_particle_force(uid);
//   if (f)
//     printf("Force: (%f, %f, %f)\n", (*f)[0], (*f)[1], (*f)[2]);
//   if (lb->is_particle_on_this_process(uid)) {
//     auto n = (*f).norm();
//     BOOST_CHECK_GT(n, 1e-3);
//   } else {
//     BOOST_CHECK(!f);
//   }

//   for (int i = 0; i < 100; ++i) {
//     lb->integrate();
//     f = lb->get_particle_force(uid);
//     auto pos = lb->get_particle_position(uid);
//     if (f)
//       printf("Force: (%f, %f, %f), Pos: (%f, %f, %f)\n", (*f)[0], (*f)[1],
//              (*f)[2], (*pos)[0], (*pos)[1], (*pos)[2]);
//   }
// }

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int n_nodes;

  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
  MPI_Dims_create(n_nodes, 3, mpi_shape.data());

  Vector3i grid_dimension{20, 20, 40};
  Vector3i block_dimension{1, 1, 1};
  bool force_avg = true;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--grid_dim") == 0) {
      grid_dimension = {std::atoi(argv[++i]), std::atoi(argv[++i]),
                        std::atoi(argv[++i])};
    }
    if (std::strcmp(argv[i], "--block_dim") == 0) {
      block_dimension = {std::atoi(argv[++i]), std::atoi(argv[++i]),
                         std::atoi(argv[++i])};
    }
    if (std::strcmp(argv[i], "--no_force_avg") == 0) {
      force_avg = false;
      ++i;
    }
  }
  params = LBTestParameters(block_dimension, grid_dimension, force_avg);
  params.particle_radius = 3;

  walberla_mpi_init();
  auto res = boost::unit_test::unit_test_main(init_unit_test, argc, argv);
  MPI_Finalize();
  return res;
}

#else  // ifdef LB_WALBERLA
int main(int argc, char **argv) {}
#endif // ifdef LB_WALBERLA
