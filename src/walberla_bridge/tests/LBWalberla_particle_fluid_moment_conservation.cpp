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

#include "field/vtk/all.h"
#include "lbm/vtk/all.h"
#include "pe/vtk/BodyVtkOutput.h"
#include "pe/vtk/SphereVtkOutput.h"
#include "stencil/D3Q27.h"
#include "vtk/all.h"

#include "utils/constants.hpp"
#include "utils/quaternion.hpp"
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <boost/qvm/vec_operations.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mpi.h>
#include <sstream>
#define BOOST_TEST_MODULE Walberla pe setters and getters test
#define BOOST_TEST_DYN_LINK
#include "config.hpp"

#ifdef LB_WALBERLA

#define BOOST_TEST_NO_MAIN

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <memory>
#include <string>
#include <vector>

#include "PE_Parameters.hpp"
#include "tests_common.hpp"
#include "utils/Vector.hpp"
#include <lb_walberla_init.hpp>

using Utils::Quaternion;
using Utils::Vector3d;
using Utils::Vector3i;

namespace bdata = boost::unit_test::data;

void write_data(uint64_t timestep, std::vector<Vector3d> vectors,
                std::string const &filename) {
  std::ofstream file;
  // file.precision(5);
  // file.setf(std::ofstream::fixed);
  file.open(filename, std::ofstream::app);

  file << timestep << "\t";
  for (auto const &vec : vectors) {
    for (uint64_t i = 0; i < 3; ++i) {
      file << vec[i] << "\t";
    }
  }

  // file << "|" << std::setw(4) << timestep << " |";
  // for (auto const &vec : vectors) {
  //   for (uint64_t i = 0; i < 3; ++i) {
  //     file << std::setw(8) << vec[i] << " ";
  //   }
  //   file << "(" << std::setw(8) << vec.norm() << ") |";
  // }
  file << std::endl;
  file.close();
}

constexpr std::uint64_t P_UID = 0;

// populated in main()

// simulation control
bool fileIO = false;
int vtkIOFreq = 0;
std::string base_folder = "data/LBWalberla_particle_fluid_moment_conservation";
std::string logging_postfix = "";
std::chrono::minutes max_simulation_minutes{5};

// numerical parameters
bool do_force_avg{false};
Vector3i grid_dimensions{54, 54, 54};
Vector3i mpi_shape;
int time_steps{20};
int reset_position_first_n_timesteps = 0;

// physical setup
double viscosity{0.1};
double density{1.};
std::vector<std::pair<Utils::Vector3d, std::string>> p_ext_forces{};
Vector3d p_init_vel{0, 0, 1e-3};
Vector3d p_init_pos{.5 * grid_dimensions};
double p_radius{7.5};
double p_density{1.16 * density};

BOOST_AUTO_TEST_CASE(momentum_conservation) {
  std::cout << "Reynolds number = " << p_init_vel[2] * 2 * p_radius / viscosity
            << std::endl;
  BOOST_CHECK_LT(p_init_vel[2] * 2 * p_radius / viscosity, 1.5);

  int n_ghost_layers{1};
  PE_Parameters pe_params{p_ext_forces, 1, true, do_force_avg};
  auto lb = walberla::LBWalberlaD3Q19MRT(viscosity, density, grid_dimensions,
                                         mpi_shape, n_ghost_layers, pe_params);

  lb.create_particle_material("Test material", p_density, 0.5, 0.1, 0.1, 0.24,
                              200, 200, 0, 0);
  lb.add_particle(P_UID, p_init_pos, p_radius, p_init_vel, "Test material");
  lb.finish_particle_adding();

  // VTK
  if (vtkIOFreq > 0) {
    unsigned flag_observables =
        static_cast<unsigned>(OutputVTK::density) |
        static_cast<unsigned>(OutputVTK::velocity_vector);
    lb.create_vtk(vtkIOFreq, 1, flag_observables, "total_mom", base_folder, "");
  }

  // Get particle attributes
  double p_mass = 0;
  if (lb.is_particle_on_this_process(P_UID)) {
    p_mass = *lb.get_particle_mass(P_UID);
  }
  MPI_Allreduce(MPI_IN_PLACE, &p_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  Vector3d const initial_momentum = p_mass * p_init_vel;

  // check fluid has no momentum
  Vector3d fluid_mom = lb.get_momentum();
  MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  BOOST_CHECK_SMALL(fluid_mom.norm(), 1e-10);

  // data file
  std::string loggingFileName(base_folder + "/additional_data");
  loggingFileName += logging_postfix;
  loggingFileName += ".txt";
  if (lb.is_particle_on_this_process(P_UID) && fileIO) {
    std::ofstream file;
    file.open(loggingFileName);
    file << "Settings: "
         << "mpi_shape=" << mpi_shape << "|"
         << "grid_dimensions=" << grid_dimensions << "|"
         << "do_force_avg=" << do_force_avg << "|"
         << "p_init_vel=" << p_init_vel << "|"
         << "p_init_pos=" << p_init_pos << "|"
         << "p_radius=" << p_radius << "|"
         << "viscosity=" << viscosity << "|"
         << "p_density=" << p_density << "|"
         << "time_steps=" << time_steps << "|"
         << "n_ghost_layers=" << n_ghost_layers << "|"
         << "density=" << density << std::endl;
    file << "#\t"
         << "fluid mom x\t"
         << "fluid mom y\t"
         << "fluid mom z\t"
         << "particle mom x\t"
         << "particle mom y\t"
         << "particle mom z\t"
         << "particle pos x\t"
         << "particle pos y\t"
         << "particle pos z\t"
         << "particle force x\t"
         << "particle force y\t"
         << "particle force z\t"
         // file << "|  #  |"
         //         "            fluid momentum            |"
         //         "           particle momentum          |"
         //         "           particle position          |"
         //         "            particle force            |"
         << std::endl;
    file.close();
  }

  Vector3d particle_mom{0, 0, 0};
  Vector3d particle_pos{};
  Vector3d particle_force{};
  Vector3d particle_ang_vel{};
  auto t0 = std::chrono::system_clock::now();
  for (uint64_t i = 0; i < time_steps; ++i) {
    if (std::chrono::system_clock::now() - t0 > max_simulation_minutes) {
      break;
    }
    if ((time_steps < 10) || (i % (time_steps / 10) == 0)) {
      if (lb.is_particle_on_this_process(P_UID))
        std::cout << "Timestep: " << i << std::endl;
    }
    fluid_mom = lb.get_momentum();
    MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    if (lb.is_particle_on_this_process(P_UID)) {
      particle_mom = *(lb.get_particle_velocity(P_UID)) * p_mass;
      particle_force = *(lb.get_particle_force(P_UID));
      particle_pos = *(lb.get_particle_position(P_UID));
      particle_ang_vel = *(lb.get_particle_angular_velocity(P_UID));

      // Check particle is far enough from boundary
      {
        auto p_ = particle_pos - .5 * grid_dimensions;
        for (int i = 0; i < 3; ++i) {
          if (p_[i] > .5 * grid_dimensions[i] - 1.5 * p_radius) {
            std::cout << "Particle too close to boundary. Stopping test..."
                      << std::endl;
            break;
          }
        }
      }

      if (fileIO) {
        write_data(i, {fluid_mom, particle_mom, particle_pos, particle_force},
                   loggingFileName);
      }
    }
    if (std::any_of(fluid_mom.begin(), fluid_mom.end(),
                    [](double a) { return std::isnan(a); })) {
      std::cout << "Found NAN values ... exiting" << std::endl;
      return;
    }
    lb.integrate();
    if (reset_position_first_n_timesteps > 0 &&
        i < reset_position_first_n_timesteps) {
      bool b = lb.set_particle_position(P_UID, p_init_pos);
      if (lb.is_particle_on_this_process(P_UID) && b)
        std::cout << "Resetting particle position ..." << std::endl;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, particle_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  // Check total momentum is conserved
  auto const measured_mom = particle_mom + fluid_mom;
  BOOST_CHECK_SMALL((initial_momentum - measured_mom).norm(), 1e-5);
}

// BOOST_AUTO_TEST_CASE(multi_momentum_conservation) {
//   int n_ghost_layers{1};
//   PE_Parameters pe_params{p_ext_forces, 1, true, do_force_avg};
//   auto lb = walberla::LBWalberlaD3Q19MRT(viscosity, density, {63, 63, 63},
//                                          mpi_shape, n_ghost_layers,
//                                          pe_params);

//   lb.create_particle_material("Test material", p_density, 0.5, 0.1, 0.1,
//   0.24,
//                               200, 200, 0, 0);
//   lb.add_particle(0, {10, 10, 10}, p_radius, p_init_vel, "Test material");
//   lb.add_particle(1, {53, 53, 53}, p_radius, -p_init_vel, "Test material");
//   lb.finish_particle_adding();

//   // VTK
//   if (vtkIOFreq > 0) {
//     unsigned flag_observables =
//         static_cast<unsigned>(OutputVTK::density) |
//         static_cast<unsigned>(OutputVTK::velocity_vector);
//     lb.create_vtk(vtkIOFreq, 1, flag_observables, "multi_total_mom",
//                   base_folder, "");
//   }

//   // Get particle attributes
//   double p_mass = 0;
//   if (lb.is_particle_on_this_process(P_UID)) {
//     p_mass = *lb.get_particle_mass(P_UID);
//   }

//   // Particle mass should be the same on all processes
//   MPI_Allreduce(MPI_IN_PLACE, &p_mass, 1, MPI_DOUBLE, MPI_SUM,
//   MPI_COMM_WORLD);

//   // check fluid has no momentum
//   Vector3d fluid_mom = lb.get_momentum();
//   MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
//                 MPI_COMM_WORLD);
//   BOOST_CHECK_SMALL(fluid_mom.norm(), 1e-10);

//   Vector3d const initial_momentum = fluid_mom;

//   Vector3d particle_mom_1{0, 0, 0};
//   Vector3d particle_mom_2{0, 0, 0};
//   auto t0 = std::chrono::system_clock::now();
//   for (uint64_t i = 0; i < time_steps; ++i) {
//     if (std::chrono::system_clock::now() - t0 > max_simulation_minutes) {
//       break;
//     }
//     if ((time_steps < 10) || (i % (time_steps / 10) == 0)) {
//       int rank;
//       MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//       if (rank == 0)
//         std::cout << "Timestep: " << i << std::endl;
//     }
//     lb.integrate();
//   }

//   fluid_mom = lb.get_momentum();
//   if (lb.is_particle_on_this_process(0))
//     particle_mom_1 = *lb.get_particle_velocity(0) * p_mass;
//   if (lb.is_particle_on_this_process(1))
//     particle_mom_2 = *lb.get_particle_velocity(1) * p_mass;
//   MPI_Allreduce(MPI_IN_PLACE, particle_mom_1.data(), 3, MPI_DOUBLE, MPI_SUM,
//                 MPI_COMM_WORLD);
//   MPI_Allreduce(MPI_IN_PLACE, particle_mom_2.data(), 3, MPI_DOUBLE, MPI_SUM,
//                 MPI_COMM_WORLD);
//   MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
//                 MPI_COMM_WORLD);
//   // Check total momentum is conserved
//   auto const measured_mom = particle_mom_1 + particle_mom_2 + fluid_mom;
//   BOOST_CHECK_SMALL((initial_momentum - measured_mom).norm(), 1e-5);
// }

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int n_nodes;

  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
  MPI_Dims_create(n_nodes, 3, mpi_shape.data());

  double acceleration = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--") == 0) { // Needed because of boost-test
      continue;
    }

    // Flags
    if (std::strcmp(argv[i], "--withForceAvg") == 0) {
      do_force_avg = true;
      continue;
    }
    if (std::strcmp(argv[i], "--fileIO") == 0) {
      fileIO = true;
      continue;
    }

    // Key-value pairs
    if (std::strcmp(argv[i], "--vtkFreq") == 0) {
      vtkIOFreq = std::atoi(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--baseFolder") == 0) {
      base_folder = argv[++i];
      continue;
    }
    if (std::strcmp(argv[i], "--logpostfix") == 0) {
      logging_postfix = argv[++i];
      continue;
    }
    if (std::strcmp(argv[i], "--minutes") == 0) {
      max_simulation_minutes = std::chrono::minutes(std::atoi(argv[++i]));
      continue;
    }
    if (std::strcmp(argv[i], "--timesteps") == 0) {
      time_steps = std::atoi(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--resolution") == 0) {
      int r = std::atoi(argv[++i]);
      grid_dimensions = {r, r, r};
      p_init_pos = .5 * grid_dimensions;
      continue;
    }
    if (std::strcmp(argv[i], "--gridDim") == 0) {
      grid_dimensions = {std::atoi(argv[++i]), std::atoi(argv[++i]),
                         std::atoi(argv[++i])};
      p_init_pos = .5 * grid_dimensions;
      continue;
    }
    if (std::strcmp(argv[i], "--resetPos") == 0) {
      reset_position_first_n_timesteps = std::atoi(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--visc") == 0) {
      viscosity = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--pdens") == 0) {
      p_density = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--radius") == 0) {
      p_radius = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--xpos") == 0) {
      p_init_pos[0] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--ypos") == 0) {
      p_init_pos[1] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--zpos") == 0) {
      p_init_pos[2] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--xvel") == 0) {
      p_init_vel[0] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--yvel") == 0) {
      p_init_vel[1] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--zvel") == 0) {
      p_init_vel[2] = std::stod(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--acc") == 0) {
      acceleration = std::atoi(argv[++i]);
      continue;
    }
    std::cout << "Unrecognized command line argument found: " << argv[i]
              << std::endl;
    return 1;
  }

  double const p_volume =
      4. / 3. * boost::math::double_constants::pi * std::pow(p_radius, 3);
  Vector3d const fz{0, 0, (p_density - density) * acceleration * p_volume};
  if (std::abs(fz[2]) > 1e-16) {
    p_ext_forces.emplace_back(fz, "");
  }

  walberla_mpi_init();
  auto res = boost::unit_test::unit_test_main(init_unit_test, argc, argv);
  MPI_Finalize();
  return res;
}

#else  // ifdef LB_WALBERLA
int main(int argc, char **argv) {}
#endif // ifdef LB_WALBERLA
