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

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(add_remove_particles) { BOOST_CHECK(1 == 2); }

#endif // ifdef LB_WALBERLA
