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

#define BOOST_TEST_MODULE Walberla settling sphere test
#define BOOST_TEST_DYN_LINK

#include "config.hpp"

#ifdef LB_WALBERLA
#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>

#include <LBWalberlaD3Q19MRT.hpp>
#include <lb_walberla_init.hpp>

#include <memory>

using Utils::Vector3d;
using Utils::Vector3i;
using uint_t = std::size_t;
using walberla::LBWalberlaD3Q19MRT;

constexpr double tau = 0.34;
Vector3i mpi_shape;

// Recreation of Walberla's SettlingSphere test
BOOST_AUTO_TEST_CASE(settling_sphere) {

  // TODO: Add those as commandline arguments
  // simulation control
  bool shortrun = false;
  bool funcTest = false;
  bool fileIO = true;
  std::string baseFolder = "vtk_out_SettlingSphere";
  uint_t fluidType = 4;

  // numerical parameters
  // uint_t numberOfCellsInHorizontalDirection = 100; // Original value
  uint_t numberOfCellsInHorizontalDirection = 90;
  bool averageForceTorqueOverTwoTimSteps = true;

  if (funcTest) {
    walberla::logging::Logging::instance()->setLogLevel(
        walberla::logging::Logging::LogLevel::WARNING);
  }

  // TODO: Change to other filesystem (Espresso / std ?)
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
  const double diameter_SI = 15e-3;
  const double densitySphere_SI = 1120;

  double densityFluid_SI, dynamicViscosityFluid_SI;
  double expectedSettlingVelocity_SI;
  switch (fluidType) {
  case 1:
    // Re_p around 1.5
    densityFluid_SI = 970;
    dynamicViscosityFluid_SI = 373e-3;
    expectedSettlingVelocity_SI = 0.035986;
    break;
  case 2:
    // Re_p around 4.1
    densityFluid_SI = 965;
    dynamicViscosityFluid_SI = 212e-3;
    expectedSettlingVelocity_SI = 0.05718;
    break;
  case 3:
    // Re_p around 11.6
    densityFluid_SI = 962;
    dynamicViscosityFluid_SI = 113e-3;
    expectedSettlingVelocity_SI = 0.087269;
    break;
  case 4:
    // Re_p around 31.9
    densityFluid_SI = 960;
    dynamicViscosityFluid_SI = 58e-3;
    expectedSettlingVelocity_SI = 0.12224;
    break;
  default:
    throw std::runtime_error(
        "Only four different fluids are supported! Choose type "
        "between 1 and 4.");
  }
  const double kinematicViscosityFluid_SI =
      dynamicViscosityFluid_SI / densityFluid_SI;

  const double gravitationalAcceleration_SI = 9.81;
  // Vector3d domainSize_SI{100e-3, 100e-3, 160e-3}; // Original values
  Vector3d domainSize_SI{90e-3, 90e-3, 150e-3};
  // shift starting gap a bit upwards to match the reported (plotted) values
  const double startingGapSize_SI = 120e-3 + 0.25 * diameter_SI;

  //////////////////////////
  // NUMERICAL PARAMETERS //
  //////////////////////////

  const double dx_SI =
      domainSize_SI[0] / double(numberOfCellsInHorizontalDirection);
  const Vector3i domainSize{int(floor(domainSize_SI[0] / dx_SI + 0.5)),
                            int(floor(domainSize_SI[1] / dx_SI + 0.5)),
                            int(floor(domainSize_SI[2] / dx_SI + 0.5))};
  const double diameter = diameter_SI / dx_SI;
  const double sphereVolume =
      walberla::math::pi / 6.0 * diameter * diameter * diameter;

  const double expectedSettlingVelocity = 0.01;
  const double dt_SI =
      expectedSettlingVelocity / expectedSettlingVelocity_SI * dx_SI;

  const double viscosity = kinematicViscosityFluid_SI * dt_SI / (dx_SI * dx_SI);
  const double relaxationTime =
      1.0 / walberla::lbm::collision_model::omegaFromViscosity(viscosity);

  const double gravitationalAcceleration =
      gravitationalAcceleration_SI * dt_SI * dt_SI / dx_SI;

  const double densityFluid = 1.0;
  const double densitySphere =
      densityFluid * densitySphere_SI / densityFluid_SI;

  const double dx = 1.0;

  const uint_t timesteps =
      funcTest ? 1 : (shortrun ? uint_t(200) : uint_t(250000));
  const uint_t numPeSubCycles = uint_t(1);

  ///////////////////////////
  // BLOCK STRUCTURE SETUP //
  ///////////////////////////

  Vector3i numberOfBlocksPerDirection{3, 3, 3};
  Vector3i cellsPerBlockPerDirection{
      domainSize[0] / numberOfBlocksPerDirection[0],
      domainSize[1] / numberOfBlocksPerDirection[1],
      domainSize[2] / numberOfBlocksPerDirection[2]};
  for (uint_t i = 0; i < 3; ++i) {
    BOOST_CHECK_MESSAGE(
        cellsPerBlockPerDirection[i] * numberOfBlocksPerDirection[i] ==
            domainSize[i],
        "Unmatching domain decomposition in direction " << i << "!");
  }

  const double overlap = 1.5 * dx;
  const bool syncShadowOwners = true;
  Vector3d initialPosition{0.5 * double(domainSize[0]),
                           0.5 * double(domainSize[1]),
                           startingGapSize_SI / dx_SI + 0.5 * diameter};

  std::string loggingFileName(baseFolder + "/LoggingSettlingSphere_");
  loggingFileName += std::to_string(fluidType);
  loggingFileName += ".txt";
  if (fileIO) {
    WALBERLA_LOG_INFO_ON_ROOT(" - writing logging output to file \""
                              << loggingFileName << "\"");
  }

  Vector3d gravitationalForce{0, 0,
                              -(densitySphere - densityFluid) *
                                  gravitationalAcceleration * sphereVolume};

  double terminationPosition =
      diameter; // right before sphere touches the bottom wall

  Vector3d linearVelocity{0, 0, 0};
  uint sphere_uid = 123;
  double agrid = dx_SI;
  const int n_ghost_layers = 1;

  walberla::PE_Parameters peParams(true, syncShadowOwners, overlap / dx,
                                   averageForceTorqueOverTwoTimSteps,
                                   numPeSubCycles);
  peParams.constantGlobalForces.push_back(
      {gravitationalForce, "Gravitational Force"});

  auto lb = std::make_shared<LBWalberlaD3Q19MRT>(
      viscosity, densityFluid, tau, numberOfBlocksPerDirection,
      cellsPerBlockPerDirection, dx_SI, mpi_shape, n_ghost_layers, peParams);

  // add the sphere
  lb->createMaterial("mySphereMat", densitySphere);
  lb->add_pe_particle(sphere_uid, initialPosition, 0.5 * diameter,
                      linearVelocity, "mySphereMat");
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
    if (i == 16) {
      WALBERLA_LOG_INFO_ON_ROOT("AUFPASSEN");
    }
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

      file << i << "\t" << double(i) * dt_SI << "\t"
           << "\t" << scaledPosition[0] << "\t" << scaledPosition[1] << "\t"
           << scaledPosition[2] - 0.5 << "\t" << velocity_SI[0] << "\t"
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
