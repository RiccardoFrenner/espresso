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

// TODO: remove
#include <core/logging/Logging.h>
#include <core/math/all.h>

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
using walberla::pe_coupling::ForceOnBodiesAdder;
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

BOOST_AUTO_TEST_CASE(add_particle_inside_domain) {
  walberla::PE_Parameters peParams(true, true, 1.5, true, 1);
  auto lb = std::make_shared<LBWalberlaD3Q19MRT>(
      viscosity, density, agrid, tau, box_dimensions, mpi_shape, 1, peParams);

  // Create particle
  std::uint64_t uid = 12;
  double radius = 0.1;
  Vector3d gpos{0, 0, 0};
  Vector3d linVel{1.0, 0.2, 0.1};
  lb->add_pe_particle(uid, gpos, radius, linVel);

  // Add force/torque
  Vector3d force{0.1, 0.5, 0.22};
  Vector3d torque{0.5, 0.1, 0.324};
  lb->set_particle_force(uid, force);
  lb->set_particle_torque(uid, torque);

  auto p = lb->get_pe_particle(uid);

  // Check that particle exists exactly on one rank
  int exists = p == nullptr ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &exists, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  BOOST_CHECK(exists == 1);

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
}

BOOST_AUTO_TEST_CASE(remove_particle) {
  walberla::PE_Parameters peParams(true, true, 1.5, true, 1);
  auto lb = std::make_shared<LBWalberlaD3Q19MRT>(
      viscosity, density, agrid, tau, box_dimensions, mpi_shape, 1, peParams);

  // Create particle
  std::uint64_t uid = 12;
  double radius = 0.1;
  Vector3d gpos{0, 0, 0};
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
  BOOST_CHECK(lb->get_pe_particle(uid) == nullptr);
}

// Recreation of the SettlingSphere test in Walberla
BOOST_AUTO_TEST_CASE(settling_sphere) {
  walberla::logging::Logging::instance()->setStreamLogLevel(
      walberla::logging::Logging::INFO);
  walberla::logging::Logging::instance()->setFileLogLevel(
      walberla::logging::Logging::TRACING);
  walberla::logging::Logging::instance()->includeLoggingToFile(
      "MEINE_LOGS_1.txt");
  WALBERLA_LOG_INFO_ON_ROOT(
      "Is logging to file enabled? -> "
      << walberla::logging::Logging::instance()->loggingToFile());

  // simulation control
  bool shortrun = false;
  bool funcTest = false;
  bool fileIO = true;
  std::string baseFolder = "vtk_out_SettlingSphere";

  // physical setup
  uint_t fluidType = 1;

  // numerical parameters
  uint_t numberOfCellsInHorizontalDirection = uint_t(100);
  bool averageForceTorqueOverTwoTimSteps = true;

  if (funcTest) {
    walberla::logging::Logging::instance()->setLogLevel(
        walberla::logging::Logging::LogLevel::WARNING);
  }

  if (fileIO) {
    // create base directory if it does not yet exist
    walberla::filesystem::path tpath(baseFolder);
    if (!walberla::filesystem::exists(tpath))
      walberla::filesystem::create_directory(tpath);
  }

  //////////////////////////////////////
  // SIMULATION PROPERTIES in SI units//
  //////////////////////////////////////

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

  WALBERLA_LOG_INFO_ON_ROOT("Setup (in SI units):");
  WALBERLA_LOG_INFO_ON_ROOT(" - domain size = " << domainSize_SI);
  WALBERLA_LOG_INFO_ON_ROOT(" - sphere: diameter = "
                            << diameter_SI << ", density = " << densitySphere_SI
                            << ", starting gap size = " << startingGapSize_SI);
  WALBERLA_LOG_INFO_ON_ROOT(" - fluid: density = "
                            << densityFluid_SI
                            << ", dyn. visc = " << dynamicViscosityFluid_SI
                            << ", kin. visc = " << kinematicViscosityFluid_SI);
  WALBERLA_LOG_INFO_ON_ROOT(" - expected settling velocity = "
                            << expectedSettlingVelocity_SI << " --> Re_p = "
                            << expectedSettlingVelocity_SI * diameter_SI /
                                   kinematicViscosityFluid_SI);

  //////////////////////////
  // NUMERICAL PARAMETERS //
  //////////////////////////

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

  WALBERLA_LOG_INFO_ON_ROOT(" - dx_SI = " << dx_SI << ", dt_SI = " << dt_SI);
  WALBERLA_LOG_INFO_ON_ROOT("Setup (in simulation, i.e. lattice, units):");
  WALBERLA_LOG_INFO_ON_ROOT(" - domain size = " << domainSize);
  WALBERLA_LOG_INFO_ON_ROOT(
      " - sphere: diameter = " << diameter << ", density = " << densitySphere);
  WALBERLA_LOG_INFO_ON_ROOT(" - fluid: density = "
                            << densityFluid << ", relaxation time (tau) = "
                            << relaxationTime << ", kin. visc = " << viscosity);
  WALBERLA_LOG_INFO_ON_ROOT(
      " - gravitational acceleration = " << gravitationalAcceleration);
  WALBERLA_LOG_INFO_ON_ROOT(" - expected settling velocity = "
                            << expectedSettlingVelocity << " --> Re_p = "
                            << expectedSettlingVelocity * diameter / viscosity);

  ///////////////////////////
  // BLOCK STRUCTURE SETUP //
  ///////////////////////////

  // Vector3<uint_t> numberOfBlocksPerDirection(uint_t(5), uint_t(5),
  // uint_t(8)); // TODO
  Vector3<uint_t> numberOfBlocksPerDirection(uint_t(1), uint_t(1), uint_t(1));
  Vector3<uint_t> cellsPerBlockPerDirection(
      domainSize[0] / numberOfBlocksPerDirection[0],
      domainSize[1] / numberOfBlocksPerDirection[1],
      domainSize[2] / numberOfBlocksPerDirection[2]);
  for (uint_t i = 0; i < 3; ++i) {
    WALBERLA_CHECK_EQUAL(
        cellsPerBlockPerDirection[i] * numberOfBlocksPerDirection[i],
        domainSize[i],
        "Unmatching domain decomposition in direction " << i << "!");
  }

  WALBERLA_LOG_INFO_ON_ROOT("Domain decomposition:");
  WALBERLA_LOG_INFO_ON_ROOT(
      " - blocks per direction = " << numberOfBlocksPerDirection);
  WALBERLA_LOG_INFO_ON_ROOT(
      " - cells per block = " << cellsPerBlockPerDirection);

  const real_t overlap = real_t(1.5) * dx;
  const bool syncShadowOwners = true;
  Vector3<real_t> initialPosition(
      real_t(0.5) * real_c(domainSize[0]), real_t(0.5) * real_c(domainSize[1]),
      startingGapSize_SI / dx_SI + real_t(0.5) * diameter);

  std::string loggingFileName(baseFolder + "/LoggingSettlingSphere_");
  loggingFileName += std::to_string(fluidType);
  loggingFileName += ".txt";
  if (fileIO) {
    WALBERLA_LOG_INFO_ON_ROOT(" - writing logging output to file \""
                              << loggingFileName << "\"");
  }

  Vector3<real_t> gravitationalForce(real_t(0), real_t(0),
                                     -(densitySphere - densityFluid) *
                                         gravitationalAcceleration *
                                         sphereVolume);

  real_t terminationPosition =
      diameter; // right before sphere touches the bottom wall

  Vector3<real_t> linearVelocity(real_t(0), real_t(0), real_t(0));
  uint sphere_uid = 123;
  double agrid = dx_SI;
  const int n_ghost_layers = 1;
  Vector3d box_dimensions = to_vector3d(domainSize_SI);

  walberla::PE_Parameters peParams(true, syncShadowOwners, overlap / dx,
                                   averageForceTorqueOverTwoTimSteps,
                                   numPeSubCycles);
  peParams.constantGlobalForces.push_back(
      {gravitationalForce, "Gravitational Force"});

  auto lb = std::make_shared<LBWalberlaD3Q19MRT>(viscosity, densityFluid, agrid,
                                                 tau, box_dimensions, mpi_shape,
                                                 n_ghost_layers, peParams);

  // add the sphere
  lb->createMaterial("mySphereMat", densitySphere);
  lb->add_pe_particle(sphere_uid, to_vector3d(initialPosition),
                      real_t(0.5) * diameter, to_vector3d(linearVelocity),
                      "mySphereMat");
  lb->finish_particle_adding();

  if (fileIO) {
    WALBERLA_ROOT_SECTION() {
      std::ofstream file;
      file.open(loggingFileName.c_str());
      file << "#\t t\t posX\t posY\t gapZ\t velX\t velY\t velZ\n";
      file.close();
    }
  }

  ////////////////////////
  // EXECUTE SIMULATION //
  ////////////////////////

  // time loop
  Vector3d vel{0, 0, 0};
  Vector3d pos{0, 0, 0};
  Vector3d force{0, 0, 0};
  double max_velocity = 0.0;
  for (uint_t i = 0; i < timesteps; ++i) {
    WALBERLA_LOG_INFO_ON_ROOT("Timestep " << i << ": ");
    // perform a single simulation step
    lb->integrate();

    // Get position and velocity of Sphere from all mpi processes
    auto p = lb->get_particle_position(sphere_uid);
    auto v = lb->get_particle_velocity(sphere_uid);
    auto f = lb->get_particle_force(sphere_uid);
    pos = p ? *p : Vector3d{0, 0, 0};
    vel = v ? *v : Vector3d{0, 0, 0};
    force = f ? *f : Vector3d{0, 0, 0};
    MPI_Allreduce(MPI_IN_PLACE, &pos[0], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pos[1], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pos[2], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &vel[0], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &vel[1], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &vel[2], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &force[0], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &force[1], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &force[2], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    max_velocity = std::max(max_velocity, -vel[2]);

    // Logging
    // ----------------------------------------------------------
    WALBERLA_ROOT_SECTION() {
      std::ofstream file;
      file.open(loggingFileName.c_str(), std::ofstream::app);

      auto scaledPosition = pos / diameter;
      auto velocity_SI = vel * dx_SI / dt_SI;

      file << i << "\t" << real_c(i) * dt_SI << "\t"
           << "\t" << scaledPosition[0] << "\t" << scaledPosition[1] << "\t"
           << scaledPosition[2] - real_t(0.5) << "\t" << velocity_SI[0] << "\t"
           << velocity_SI[1] << "\t" << velocity_SI[2] << "\t" << force[0]
           << "\t" << force[1] << "\t" << force[2] << "\n";
      file.close();
    }
    // ----------------------------------------------------------

    if (pos[2] < terminationPosition) {
      WALBERLA_LOG_INFO_ON_ROOT("Sphere reached terminal position "
                                << pos[2] << " after " << i << " timesteps!");
      break;
    }
  }

  // check the result
  if (!funcTest && !shortrun) {
    double relErr = std::fabs(expectedSettlingVelocity - max_velocity) /
                    expectedSettlingVelocity;
    WALBERLA_LOG_INFO_ON_ROOT(
        "Expected maximum settling velocity: " << expectedSettlingVelocity);
    WALBERLA_LOG_INFO_ON_ROOT(
        "Simulated maximum settling velocity: " << max_velocity);
    WALBERLA_LOG_INFO_ON_ROOT("Relative error: " << relErr);

    // the relative error has to be below 10%
    BOOST_CHECK(relErr < 0.1);
  }
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
