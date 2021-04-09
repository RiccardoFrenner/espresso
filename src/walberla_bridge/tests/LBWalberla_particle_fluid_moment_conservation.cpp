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

#include "PE_Parameters.hpp"
#include "utils/Vector.hpp"
#include "utils/constants.hpp"
#include <cmath>
#define BOOST_TEST_MODULE Walberla particle fluid moment conservation
#define BOOST_TEST_DYN_LINK

#include "config.hpp"

#ifdef LB_WALBERLA
#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>

#include <LBWalberlaD3Q19MRT.hpp>
#include <lb_walberla_init.hpp>

using Utils::Vector3d;
using Utils::Vector3i;

constexpr double AGRID = .4;
// constexpr double DENS = 2.3;
// constexpr double KVISC = 4.;
constexpr double DENS = 970.;
constexpr double KVISC = 373e-3 / DENS;
constexpr double TIME_STEP = 0.008;
constexpr int GRID_SIZE = 6;

// Particle data
// constexpr double PARTICLE_RADIUS = 3 * AGRID;
constexpr double PARTICLE_RADIUS = 15e-3 / AGRID;
// constexpr double PARTICLE_DENSITY = 1. / PARTICLE_VOLUME;
constexpr double PARTICLE_DENSITY = 1120. / DENS;


constexpr Vector3d BOX_DIMENSIONS{GRID_SIZE * AGRID, GRID_SIZE *AGRID,
                                  GRID_SIZE *AGRID};
constexpr double SYSTEM_VOLUME =
    GRID_SIZE * AGRID * GRID_SIZE * AGRID * GRID_SIZE * AGRID;

constexpr double PARTICLE_VOLUME =
    4. / 3. * Utils::pi() * PARTICLE_RADIUS * PARTICLE_RADIUS * PARTICLE_RADIUS;
constexpr double PARTICLE_MASS = PARTICLE_DENSITY * PARTICLE_VOLUME;

Vector3i mpi_shape;

BOOST_AUTO_TEST_CASE(external_force) {
  constexpr double F = 5.5 / double(GRID_SIZE * GRID_SIZE * GRID_SIZE);
  constexpr Vector3d EXT_FORCE_DENSITY{-.7 * F, .9 * F, .8 * F};
  PE_Parameters pe_params(true, true, 1.5, true, 1);
  pe_params.add_global_constant_force(-SYSTEM_VOLUME * EXT_FORCE_DENSITY,
                                      "External force");
  auto lbf = new_lb_walberla(KVISC, DENS, AGRID, TIME_STEP, BOX_DIMENSIONS,
                             mpi_shape, 0, 0, pe_params);
  lbf->create_particle_material("myMat", PARTICLE_DENSITY, .5, .1, .1, .24, 200,
                                200, 0, 0);
  lbf->add_particle(42, BOX_DIMENSIONS / 2, PARTICLE_RADIUS,
                    Vector3d{.2, .4, .6}, "myMat");

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 500; ++j) {
      lbf->integrate();
    }
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