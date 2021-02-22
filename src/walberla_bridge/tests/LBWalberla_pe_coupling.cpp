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
#include <core/math/all.h> // TODO: remove
#include <lb_walberla_init.hpp>
#include <walberla_utils.hpp>

#include <memory>

using Utils::Vector3d;
using Utils::Vector3i;
using walberla::LBWalberlaD3Q19MRT;
using walberla::real_c;
using walberla::real_t;
using walberla::to_vector3;
using walberla::to_vector3d;
using walberla::uint_c;
using walberla::Vector3;
using uint_t = std::size_t; // TODO: replace all uints with size_t
// TODO: replace all walberla types with espresso types

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

// Recreation of the SettlingSphere test in Walberla
BOOST_AUTO_TEST_CASE(integration) {
  auto lb = std::make_shared<LBWalberlaD3Q19MRT>(viscosity, density, agrid, tau,
                                                 box_dimensions, mpi_shape, 1);

  // simulation control
  bool shortrun = false;
  bool funcTest = false;
  bool fileIO = false;

  // physical setup
  uint_t fluidType = 1;

  // values are mainly taken from the reference paper
  const real_t diameter_SI = real_t(15e-3);
  const real_t densitySphere_SI = real_t(1120);

  real_t densityFluid_SI, dynamicViscosityFluid_SI;
  real_t expectedSettlingVelocity_SI;
  switch (fluidType) {
  case 1:
    // Re_p around 1.5
    densityFluid_SI = real_t(970);
    dynamicViscosityFluid_SI = real_t(373e-3);
    expectedSettlingVelocity_SI = real_t(0.035986);
    break;
  case 2:
    // Re_p around 4.1
    densityFluid_SI = real_t(965);
    dynamicViscosityFluid_SI = real_t(212e-3);
    expectedSettlingVelocity_SI = real_t(0.05718);
    break;
  case 3:
    // Re_p around 11.6
    densityFluid_SI = real_t(962);
    dynamicViscosityFluid_SI = real_t(113e-3);
    expectedSettlingVelocity_SI = real_t(0.087269);
    break;
  case 4:
    // Re_p around 31.9
    densityFluid_SI = real_t(960);
    dynamicViscosityFluid_SI = real_t(58e-3);
    expectedSettlingVelocity_SI = real_t(0.12224);
    break;
  default:
    WALBERLA_ABORT("Only four different fluids are supported! Choose type "
                   "between 1 and 4.");
  }

  const real_t kinematicViscosityFluid_SI =
      dynamicViscosityFluid_SI / densityFluid_SI;

  const real_t gravitationalAcceleration_SI = real_t(9.81);
  Vector3<real_t> domainSize_SI(real_t(100e-3), real_t(100e-3), real_t(160e-3));
  // shift starting gap a bit upwards to match the reported (plotted) values
  const real_t startingGapSize_SI = real_t(120e-3) + real_t(0.25) * diameter_SI;

  //////////////////////////
  // NUMERICAL PARAMETERS //
  //////////////////////////
  uint_t numberOfCellsInHorizontalDirection = uint_t(100);
  const real_t dx_SI =
      domainSize_SI[0] / real_c(numberOfCellsInHorizontalDirection);
  const Vector3<uint_t> domainSize(
      uint_c(floor(domainSize_SI[0] / dx_SI + real_t(0.5))),
      uint_c(floor(domainSize_SI[1] / dx_SI + real_t(0.5))),
      uint_c(floor(domainSize_SI[2] / dx_SI + real_t(0.5))));
  const real_t diameter = diameter_SI / dx_SI;
  const real_t sphereVolume =
      walberla::math::pi / real_t(6) * diameter * diameter * diameter;

  const real_t expectedSettlingVelocity = real_t(0.01);
  const real_t dt_SI =
      expectedSettlingVelocity / expectedSettlingVelocity_SI * dx_SI;

  const real_t viscosity = kinematicViscosityFluid_SI * dt_SI / (dx_SI * dx_SI);
  const real_t relaxationTime =
      real_t(1) / walberla::lbm::collision_model::omegaFromViscosity(viscosity);

  const real_t gravitationalAcceleration =
      gravitationalAcceleration_SI * dt_SI * dt_SI / dx_SI;

  const real_t densityFluid = real_t(1);
  const real_t densitySphere =
      densityFluid * densitySphere_SI / densityFluid_SI;

  const real_t dx = real_t(1);

  const uint_t timesteps =
      funcTest ? 1 : (shortrun ? uint_t(200) : uint_t(250000));
  const uint_t numPeSubCycles = uint_t(1);

  const real_t densityFluid = real_t(1);
  const real_t densitySphere =
      densityFluid * densitySphere_SI / densityFluid_SI;
  // add the sphere
  lb->createMaterial("mySphereMat", densitySphere, real_t(0.5), real_t(0.1),
                     real_t(0.1), real_t(0.24), real_t(200), real_t(200),
                     real_t(0), real_t(0));
  Vector3<real_t> initialPosition(
      real_t(0.5) * real_c(domainSize[0]), real_t(0.5) * real_c(domainSize[1]),
      startingGapSize_SI / dx_SI + real_t(0.5) * diameter);
  Vector3<real_t> linearVelocity(real_t(0), real_t(0), real_t(0));
  uint uid = 123;
  lb->add_pe_particle(uid, to_vector3d(initialPosition), real_t(0.5) * diameter,
                      to_vector3d(linearVelocity), "mySphereMat");
  lb->sync_pe_particles();
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
