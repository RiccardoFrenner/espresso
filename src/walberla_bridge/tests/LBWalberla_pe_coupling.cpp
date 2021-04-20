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
  BOOST_CHECK(lb->is_particle_on_this_process(uid) == false);
}

// Check that particle receives correct hydrodynamik force
BOOST_DATA_TEST_CASE(MEM_forces, bdata::make(pe_enabled_lbs()), lb_generator) {
  auto lb = lb_generator(mpi_shape, params);
}

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

BOOST_DATA_TEST_CASE(energy_conservation, bdata::make(pe_enabled_lbs()),
                     lb_generator) {
  params.particle_initial_velocity = Vector3d{.1, 0, 0};
  auto lb = lb_generator(mpi_shape, params);

  // check fluid has no momentum
  Vector3d fluid_mom = lb->get_momentum();
  MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  BOOST_CHECK_SMALL(fluid_mom.norm(), 1e-10);

  constexpr uint64_t uid = 0;
  constexpr uint64_t steps = 100;

  double initial_particle_energy = .5 * params.get_particle_mass() *
                                   Utils::dot(params.particle_initial_velocity,
                                              params.particle_initial_velocity);
  double initial_fluid_energy = lb->get_energy();
  double particle_energy = 0.;
  double fluid_energy = 0.;
  Vector3d particle_lin_velocity{};
  Vector3d particle_avg_velocity{};
  for (uint64_t i = 0; i < steps; ++i) {
    lb->integrate();
  }
  fluid_energy = lb->get_energy();
  MPI_Allreduce(MPI_IN_PLACE, &fluid_energy, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  if (lb->is_particle_on_this_process(uid)) {
    particle_lin_velocity = *(lb->get_particle_velocity(uid));
    particle_avg_velocity = *(lb->get_particle_angular_velocity(uid));
    particle_energy = .5 * params.get_particle_mass() *
                      Utils::dot(particle_lin_velocity, particle_lin_velocity);
    BOOST_CHECK_SMALL(particle_avg_velocity.norm(), 1e-8);
    BOOST_CHECK_CLOSE(initial_particle_energy + initial_fluid_energy,
                      particle_energy + fluid_energy, 1e-4);
  }
}

BOOST_DATA_TEST_CASE(bb_boundary, bdata::make(pe_enabled_lbs()), lb_generator) {
  params.particle_initial_velocity = Vector3d{.011, -.013, .021};
  auto lb = lb_generator(mpi_shape, params);

  // check fluid has no momentum
  Vector3d fluid_mom = lb->get_momentum();
  MPI_Allreduce(MPI_IN_PLACE, fluid_mom.data(), 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  BOOST_CHECK_SMALL(fluid_mom.norm(), 1e-10);

  constexpr uint64_t uid = 0;
  constexpr uint64_t steps = 100;

  Vector3d p_surf_point = params.particle_initial_position +
                          params.particle_radius * Vector3d{1, 0, 0};

  // check that we are at a particle-fluid transition
  Vector3i p_surf_node{int(p_surf_point[0] - .5), int(p_surf_point[1] - .5),
                       int(p_surf_point[2] - .5)};
  auto is_boundary_left = lb->get_node_is_boundary(p_surf_node);
  auto is_boundary_right =
      lb->get_node_is_boundary(p_surf_node + Vector3i{1, 0, 0});
  if (is_boundary_left) {
    BOOST_CHECK(*is_boundary_left == true);
  }
  if (is_boundary_right) {
    BOOST_CHECK(*is_boundary_right == false);
  }

  // get velocity at particle surface
  Vector3d p_rot_vel{0, 0, 0};
  auto d = p_surf_point - params.particle_initial_position;
  auto p_surf_vel =
      params.particle_initial_velocity + boost::qvm::cross(p_rot_vel, d);

  // check population changes correctly
  using Stencil = walberla::stencil::D3Q19;
  auto fluid_to_boundary_dir = walberla::stencil::directionFromAxis(0, false);
  auto pdf_old = lb->get_node_pop(p_surf_node + Vector3i{1, 0, 0});
  double pdf_expected = 0;
  if (pdf_old) {
    double lattice_weight =
        walberla::lbm::MRTLatticeModel::w[Stencil::idx[fluid_to_boundary_dir]];
    double lattice_speed_of_sound = 1. / std::sqrt(3.);
    pdf_expected =
        (*pdf_old)[Stencil::idx[fluid_to_boundary_dir]] -
        2 * lattice_weight / lattice_speed_of_sound * params.density *
            (p_surf_vel[0] *
                 walberla::stencil::cx[Stencil::idx[fluid_to_boundary_dir]] +
             p_surf_vel[1] *
                 walberla::stencil::cy[Stencil::idx[fluid_to_boundary_dir]] +
             p_surf_vel[2] *
                 walberla::stencil::cz[Stencil::idx[fluid_to_boundary_dir]]);
  }
  lb->integrate();
  auto pdf_new = lb->get_node_pop(p_surf_node + Vector3i{1, 0, 0});
  if (pdf_new) {
    BOOST_CHECK_CLOSE(
        (*pdf_new)[Stencil::invDirIdx(fluid_to_boundary_dir)], pdf_expected,
        1e-6); // fails. Maybe because surronding cells add nonzero values
               // because they also interact with particle?
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int n_nodes;

  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
  MPI_Dims_create(n_nodes, 3, mpi_shape.data());

  Vector3i grid_dimension{24, 24, 24};
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
