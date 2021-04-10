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

Vector3i mpi_shape;

// Recreation of Walberla's SettlingSphere test
BOOST_AUTO_TEST_CASE(settling_sphere) {

  // TODO: Add those as commandline arguments
  // simulation control
  bool shortrun = false;
  bool func_test = false;
  bool file_io = true;
  std::string base_folder = "vtk_out_SettlingSphere";
  uint_t fluid_type = 4;

  // numerical parameters
  // uint_t n_horizontal_cells = 100; // Original value
  uint_t n_horizontal_cells = 90;
  bool average_force_torque_over_two_timesteps = true;

  if (func_test) {
    walberla::logging::Logging::instance()->setLogLevel(
        walberla::logging::Logging::LogLevel::WARNING);
  }

  // TODO: Change to other filesystem (Espresso / std ?)
  if (file_io) {
    // create base directory if it does not yet exist
    walberla::filesystem::path tpath(base_folder);
    if (!walberla::filesystem::exists(tpath))
      walberla::filesystem::create_directory(tpath);
  }

  //////////////////////////////////////
  // SIMULATION PROPERTIES in SI units//
  //////////////////////////////////////

  // values are mainly taken from the reference paper
  const double diameter_SI = 15e-3;
  const double density_sphere_SI = 1120;

  double density_fluid_SI, dynamic_viscosity_fluid_SI;
  double expected_settling_velocity_SI;
  switch (fluid_type) {
  case 1:
    // Re_p around 1.5
    density_fluid_SI = 970;
    dynamic_viscosity_fluid_SI = 373e-3;
    expected_settling_velocity_SI = 0.035986;
    break;
  case 2:
    // Re_p around 4.1
    density_fluid_SI = 965;
    dynamic_viscosity_fluid_SI = 212e-3;
    expected_settling_velocity_SI = 0.05718;
    break;
  case 3:
    // Re_p around 11.6
    density_fluid_SI = 962;
    dynamic_viscosity_fluid_SI = 113e-3;
    expected_settling_velocity_SI = 0.087269;
    break;
  case 4:
    // Re_p around 31.9
    density_fluid_SI = 960;
    dynamic_viscosity_fluid_SI = 58e-3;
    expected_settling_velocity_SI = 0.12224;
    break;
  default:
    throw std::runtime_error(
        "Only four different fluids are supported! Choose type "
        "between 1 and 4.");
  }
  const double kinematic_viscosity_fluid_SI =
      dynamic_viscosity_fluid_SI / density_fluid_SI;

  const double gravitational_acceleration_SI = 9.81;
  // Vector3d domain_size_SI{100e-3, 100e-3, 160e-3}; // Original values
  Vector3d domain_size_SI{90e-3, 90e-3, 150e-3};
  // shift starting gap a bit upwards to match the reported (plotted) values
  const double starting_gap_size_SI = 120e-3 + 0.25 * diameter_SI;

  //////////////////////////
  // NUMERICAL PARAMETERS //
  //////////////////////////

  const double dx_SI = domain_size_SI[0] / double(n_horizontal_cells);
  const Vector3i domain_size{int(floor(domain_size_SI[0] / dx_SI + 0.5)),
                             int(floor(domain_size_SI[1] / dx_SI + 0.5)),
                             int(floor(domain_size_SI[2] / dx_SI + 0.5))};
  const double diameter = diameter_SI / dx_SI;
  const double sphere_volume =
      walberla::math::pi / 6.0 * diameter * diameter * diameter;

  const double expected_settling_velocity = 0.01;
  const double dt_SI =
      expected_settling_velocity / expected_settling_velocity_SI * dx_SI;

  const double viscosity =
      kinematic_viscosity_fluid_SI * dt_SI / (dx_SI * dx_SI);
  const double relaxation_time =
      1.0 / walberla::lbm::collision_model::omegaFromViscosity(viscosity);

  const double gravitational_acceleration =
      gravitational_acceleration_SI * dt_SI * dt_SI / dx_SI;

  const double density_fluid = 1.0;
  const double density_sphere =
      density_fluid * density_sphere_SI / density_fluid_SI;

  const double dx = 1.0;

  const uint_t timesteps = func_test ? 1 : (shortrun ? 200 : 250000);
  const uint_t num_pe_sub_cycles = 1;

  ///////////////////////////
  // BLOCK STRUCTURE SETUP //
  ///////////////////////////

  Vector3i n_blocks_per_direction{3, 3, 3};
  Vector3i n_cells_per_block_per_direction{
      domain_size[0] / n_blocks_per_direction[0],
      domain_size[1] / n_blocks_per_direction[1],
      domain_size[2] / n_blocks_per_direction[2]};
  for (uint_t i = 0; i < 3; ++i) {
    BOOST_CHECK_MESSAGE(
        n_cells_per_block_per_direction[i] * n_blocks_per_direction[i] ==
            domain_size[i],
        "Unmatching domain decomposition in direction " << i << "!");
  }

  const double overlap = 1.5 * dx;
  const bool sync_shadow_owners = true;
  Vector3d initial_position{0.5 * double(domain_size[0]),
                            0.5 * double(domain_size[1]),
                            starting_gap_size_SI / dx_SI + 0.5 * diameter};

  std::string logging_file_name(base_folder + "/LoggingSettlingSphere_");
  logging_file_name += std::to_string(fluid_type);
  logging_file_name += ".txt";
  if (file_io) {
    WALBERLA_LOG_INFO_ON_ROOT(" - writing logging output to file \""
                              << logging_file_name << "\"");
  }

  Vector3d gravitational_force{0, 0,
                               -(density_sphere - density_fluid) *
                                   gravitational_acceleration * sphere_volume};

  double termination_position =
      diameter; // right before sphere touches the bottom wall

  Vector3d linear_velocity{0, 0, 0};
  uint sphere_uid = 123;
  const int n_ghost_layers = 1;

  PE_Parameters pe_params(true, sync_shadow_owners, overlap / dx,
                          average_force_torque_over_two_timesteps,
                          num_pe_sub_cycles);
  pe_params.constant_global_forces.emplace_back(
      std::make_pair(gravitational_force, "Gravitational Force"));

  auto lb = std::make_shared<LBWalberlaD3Q19MRT>(
      viscosity, density_fluid, n_blocks_per_direction,
      n_cells_per_block_per_direction, mpi_shape, n_ghost_layers,
      pe_params);

  // add the sphere
  lb->create_particle_material("mySphereMat", density_sphere, 0.5, 0.1, 0.1,
                               0.24, 200, 200, 0, 0);
  lb->add_particle(sphere_uid, initial_position, 0.5 * diameter,
                   linear_velocity, "mySphereMat");
  lb->finish_particle_adding();

  if (file_io) {
    WALBERLA_ROOT_SECTION() {
      std::ofstream file;
      file.open(logging_file_name.c_str());
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
      file.open(logging_file_name.c_str(), std::ofstream::app);

      auto scaled_position = pos / diameter;
      auto velocity_SI = vel * dx_SI / dt_SI;

      file << i << "\t" << double(i) * dt_SI << "\t"
           << "\t" << scaled_position[0] << "\t" << scaled_position[1] << "\t"
           << scaled_position[2] - 0.5 << "\t" << velocity_SI[0] << "\t"
           << velocity_SI[1] << "\t" << velocity_SI[2] << "\t" << force[0]
           << "\t" << force[1] << "\t" << force[2] << "\n";
      file.close();
    }
    // ----------------------------------------------------------

    if (pos[2] < termination_position) {
      WALBERLA_LOG_INFO_ON_ROOT("Sphere reached terminal position "
                                << pos[2] << " after " << i << " timesteps!");
      break;
    }
  }

  // check the result
  if (!func_test && !shortrun) {
    double rel_err = std::fabs(expected_settling_velocity - max_velocity) /
                     expected_settling_velocity;
    WALBERLA_LOG_INFO_ON_ROOT(
        "Expected maximum settling velocity: " << expected_settling_velocity);
    WALBERLA_LOG_INFO_ON_ROOT(
        "Simulated maximum settling velocity: " << max_velocity);
    WALBERLA_LOG_INFO_ON_ROOT("Relative error: " << rel_err);

    // the relative error has to be below 10%
    BOOST_CHECK(rel_err < 0.1);
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
