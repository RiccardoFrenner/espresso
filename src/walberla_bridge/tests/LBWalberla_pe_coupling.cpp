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

#define BOOST_TEST_MODULE Walberla pe setters and getters test
#define BOOST_TEST_DYN_LINK

#include "config.hpp"

#ifdef LB_WALBERLA
#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>

#include <LBWalberlaD3Q19MRT.hpp>
#include <lb_walberla_init.hpp>
#include <walberla_utils.hpp>

#include <memory>

using Utils::Vector3d;
using Utils::Vector3i;
using walberla::LBWalberlaD3Q19MRT;
using walberla::to_vector3;
using walberla::to_vector3d;

constexpr double viscosity = 0.4;
constexpr Vector3d box_dimensions = {6, 6, 9};
constexpr double agrid = 0.5;
constexpr Vector3i grid_dimensions{int(box_dimensions[0] / agrid),
                                   int(box_dimensions[1] / agrid),
                                   int(box_dimensions[2] / agrid)};
constexpr double tau = 0.34;
constexpr double density = 2.5;
Vector3i mpi_shape;
constexpr unsigned int seed = 3;
constexpr double kT = 0.0014;

BOOST_AUTO_TEST_CASE(add_particle) {
  auto lb = std::make_shared<LBWalberlaD3Q19MRT>(viscosity, density, agrid, tau,
                                                 box_dimensions, mpi_shape, 1);

  // Create particle
  std::uint64_t uid = 12;
  double radius = 0.1;
  Vector3d gpos{123, 456, 789};
  Vector3d linVel{1.0, 0.2, 0.1};
  lb->add_pe_particle(uid, gpos, radius, linVel);

  // Add force/torque
  Vector3d force{0.1, 0.5, 0.22};
  Vector3d torque{0.5, 0.1, 0.324};
  lb->set_particle_force(uid, force);
  lb->set_particle_torque(uid, torque);

  auto p = lb->get_pe_particle(uid);

  // Check if p is on this rank
  if (p != nullptr) {
    BOOST_CHECK(static_cast<uint64_t>(p->getID()) == uid);
    BOOST_CHECK(lb->get_particle_position(uid) == gpos);
    BOOST_CHECK(lb->get_particle_velocity(uid) == linVel);
    BOOST_CHECK(to_vector3d(p->getForce()) == force);
    BOOST_CHECK(to_vector3d(p->getTorque()) == torque);
    // BOOST_CHECK(lb->get_particle_angular_velocity() == angularVel); // TODO
    // BOOST_CHECK(lb->get_particle_orientation() == orientation); // TODO
  } else {
    // Check that access to particle attributes is not possible
    BOOST_CHECK(!lb->get_particle_position(uid));
    BOOST_CHECK(!lb->get_particle_velocity(uid));
  }

  // Check wrong uid does not exist
  uint64_t wrong_uid = 99;
  BOOST_CHECK(!lb->get_pe_particle(wrong_uid));
}

BOOST_AUTO_TEST_CASE(remove_particle) {
  auto lb = std::make_shared<LBWalberlaD3Q19MRT>(viscosity, density, agrid, tau,
                                                 box_dimensions, mpi_shape, 1);

  // Create particle
  std::uint64_t uid = 12;
  double radius = 0.1;
  Vector3d gpos{123, 456, 789};
  Vector3d linVel{1.0, 0.2, 0.1};
  lb->add_pe_particle(uid, gpos, radius, linVel);

  // Add force/torque
  Vector3d force{0.1, 0.5, 0.22};
  Vector3d torque{0.5, 0.1, 0.324};
  lb->set_particle_force(uid, force);
  lb->set_particle_torque(uid, torque);

  auto p = lb->get_pe_particle(uid);

  lb->remove_pe_particle(uid);

  // Check that particle doesn't exist on any rank
  BOOST_CHECK(p == nullptr);
  BOOST_CHECK(lb->get_pe_particle(uid) == nullptr);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int n_nodes;

  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
  MPI_Dims_create(n_nodes, 3, mpi_shape.data());

  walberla_mpi_init();
  auto res = boost::unit_test::unit_test_main(init_unit_test, argc, argv);
  MPI_Finalize();
  return res;
}

#else  // ifdef LB_WALBERLA
int main(int argc, char **argv) {}
#endif // ifdef LB_WALBERLA
