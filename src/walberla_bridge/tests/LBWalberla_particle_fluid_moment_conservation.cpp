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

#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE Walberla pe coupling
#define BOOST_TEST_DYN_LINK
#include "config.hpp"

#ifdef LB_WALBERLA

#define BOOST_TEST_NO_MAIN

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <LBWalberlaBase.hpp>
#include <lb_walberla_init.hpp>

#include <utils/Vector.hpp>

#include <mpi.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "tests_common.hpp"

using Utils::hadamard_product;
using Utils::Vector3d;
using Utils::Vector3i;

namespace bdata = boost::unit_test::data;

LBTestParameters params; // populated in main()
Vector3i mpi_shape;      // populated in main

// BOOST_DATA_TEST_CASE(momentum_zero, bdata::make(pe_enabled_lbs()),
//                      lb_generator) {
//   auto lb = lb_generator(mpi_shape, params);
//   // create particle without velocity
//   lb->create_particle_material("myMat", 1.16667, 0.5, 0.1, 0.1, 0.24, 200,
//   200,
//                                0, 0);
//   std::uint64_t const uid = 42;
//   lb->add_particle(uid, {1.5, 1.5, 1.5}, 7.5, {0, 0, 0}, "myMat");
//   lb->finish_particle_adding();
//   // check total momentum = 0
//   auto fluid_mom = lb->get_momentum();
//   MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
//                 MPI_COMM_WORLD);
//   Vector3d particle_mom{};
//   if (lb->is_particle_on_this_process(uid)) {
//     auto v = *(lb->get_particle_velocity(uid));
//     auto m = *(lb->get_particle_mass(uid));
//     particle_mom = v * m;
//     BOOST_CHECK_SMALL((fluid_mom + particle_mom).norm(), 1E-10);
//   }

//   // total momentum still zero after some timesteps
//   int const steps = 20;
//   for (int i = 0; i < steps; ++i)
//     lb->integrate();
//   fluid_mom = lb->get_momentum();
//   MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
//                 MPI_COMM_WORLD);
//   particle_mom.fill(0.);
//   if (lb->is_particle_on_this_process(uid)) {
//     auto v = *(lb->get_particle_velocity(uid));
//     auto m = *(lb->get_particle_mass(uid));
//     printf("fm: (%f, %f, %f)\n", fluid_mom[0], fluid_mom[1], fluid_mom[2]);
//     printf("v : (%f, %f, %f)\n", v[0], v[1], v[2]);
//     printf("m :  %f\n", m);
//     particle_mom = v * m;
//     BOOST_CHECK_SMALL((fluid_mom + particle_mom).norm(), 1E-10);
//   }
// }

BOOST_DATA_TEST_CASE(momentum_conservation, bdata::make(pe_enabled_lbs()),
                     lb_generator) {
  auto lb = lb_generator(mpi_shape, params);
  // create particle with velocity
  lb->create_particle_material("myMat2", 1.16667, 0.5, 0.1, 0.1, 0.24, 200, 200,
                               0, 0);
  Vector3d vel{.1, .2, .3};
  std::uint64_t const uid = 42;
  lb->add_particle(uid, {10., 10., 10.}, 7.5, vel, "myMat2");
  lb->finish_particle_adding();
  // check fluid momentum = 0
  auto fluid_mom = lb->get_momentum();
  MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  BOOST_CHECK_SMALL(fluid_mom.norm(), 1E-10);

  // momentum conservation after some timesteps
  int const steps = 20;
  for (int i = 0; i < steps; ++i) {
    lb->integrate();
    printf("i = %d\n", i);
  }
  fluid_mom = lb->get_momentum();
  MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  Vector3d particle_mom{};
  Vector3d initial_mom{};
  if (lb->is_particle_on_this_process(uid)) {
    auto v = *(lb->get_particle_velocity(uid));
    auto m = *(lb->get_particle_mass(uid));
    printf("fm: (%f, %f, %f)\n", fluid_mom[0], fluid_mom[1], fluid_mom[2]);
    printf("v : (%f, %f, %f)\n", v[0], v[1], v[2]);
    printf("m :  %f\n", m);
    particle_mom = v * m;
    initial_mom = vel * m;
    BOOST_CHECK_SMALL((fluid_mom + particle_mom - initial_mom).norm(), 1E-10);
  }
}

// BOOST_DATA_TEST_CASE(no_particle, bdata::make(pe_enabled_lbs()),
// lb_generator) {
//   auto lb = lb_generator(mpi_shape, params);
//   // check fluid momentum = 0
//   auto fluid_mom = lb->get_momentum();
//   MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
//                 MPI_COMM_WORLD);
//   BOOST_CHECK_SMALL(fluid_mom.norm(), 1E-10);

//   // momentum conservation after some timesteps
//   int const steps = 20;
//   for (int i = 0; i < steps; ++i)
//     lb->integrate();
//   fluid_mom = lb->get_momentum();
//   MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
//                 MPI_COMM_WORLD);
//   printf("fm: (%f, %f, %f)\n", fluid_mom[0], fluid_mom[1], fluid_mom[2]);

//   BOOST_CHECK_SMALL(fluid_mom.norm(), 1E-10);
// }

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int n_nodes;

  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
  MPI_Dims_create(n_nodes, 3, mpi_shape.data());

  // params.viscosity = 0.003;
  // params.kT = 1.3E-4;
  // params.density = 1.4;
  // params.grid_dimensions = Vector3i{12, 12, 18};
  // params.box_dimensions = Vector3d{12, 12, 18};
  params.viscosity = 0.00494246;
  params.kT = 0.;
  params.density = 1.0;
  params.grid_dimensions = Vector3i{100, 100, 160};
  // params.box_dimensions = Vector3d{12, 12, 18};

  walberla_mpi_init();
  auto res = boost::unit_test::unit_test_main(init_unit_test, argc, argv);
  MPI_Finalize();
  return res;
}

#else // ifdef LB_WALBERLA
int main(int argc, char **argv) {}
#endif
