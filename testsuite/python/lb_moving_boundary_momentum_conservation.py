# Copyright (C) 2010-2019 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import espressomd
import unittest as ut

import unittest_decorators as utx

import numpy as np
from itertools import count

# Define the LB Parameters
TIME_STEP = 1 #0.008
AGRID = 1 #.4
GRID_SIZE = 54
KVISC = 0.1 * AGRID**2 / TIME_STEP
DENS = 1 * AGRID**(-3)
P_UID = 0
P_DENSITY = 1.1 * DENS
P_RADIUS = 7.5 * AGRID

F = 5.5 / GRID_SIZE**3
EXTERNAL_FORCE_DENSITY = np.array([-.7 * F, .9 * F, .8 * F])
GAMMA = 1

P_VOLUME = 4./3. * np.pi * np.power(P_RADIUS,3)
P_MASS = P_VOLUME * P_DENSITY
SYSTEM_VOLUME = (AGRID*GRID_SIZE)**3
EXTERNAL_FLUID_FORCE = EXTERNAL_FORCE_DENSITY * (SYSTEM_VOLUME - P_VOLUME)
EXT_PARTICLE_FORCE = - EXTERNAL_FLUID_FORCE

class Momentum(object):
    """
    Tests momentum conservation for an LB coupled to an extended particle.
    """
    lbf = None
    system = espressomd.System(box_l=[GRID_SIZE * AGRID] * 3)
    system.time_step = TIME_STEP
    system.cell_system.skin = 0.01

    def test_init_velocity(self):
        """
        Tests momentum conservation where the particle has an initial velocity.
        """
        LB_PARAMS = {
            'agrid': AGRID,
             'dens': DENS,
             'visc': KVISC,
             'tau': TIME_STEP,
             "pe_params": ([],)
        }
        self.lbf = espressomd.lb.LBFluidWalberla(**LB_PARAMS)
        self.system.actors.add(self.lbf)
        self.system.thermostat.set_lb(LB_fluid=self.lbf, gamma=GAMMA, seed=1)

        # Initial momentum before integration = 0
        np.testing.assert_allclose(
            self.system.analysis.linear_momentum(), [0., 0., 0.], atol=1E-12)

        # Add walberla particle
        p_initial_v = np.array([0 ,0, 1e-3*AGRID])
        self.lbf.create_particle_material("myMat", P_DENSITY)
        self.lbf.add_particle(P_UID, self.system.box_l/2, P_RADIUS, p_initial_v, "myMat")
        self.lbf.finish_particle_adding()

        p_pos = self.lbf.get_particle_position(P_UID)
        p_v = self.lbf.get_particle_velocity(P_UID)

        # Check particle attributes
        np.testing.assert_allclose(np.copy(p_pos), np.copy(self.system.box_l/2))
        np.testing.assert_allclose(np.copy(p_v), p_initial_v)

        get_fluid_mom = lambda : np.array(
            self.system.analysis.linear_momentum(include_particles=False))

        initial_momentum = get_fluid_mom() + p_v * P_MASS
        measured_momentum = get_fluid_mom() + p_v * P_MASS
        np.testing.assert_allclose(initial_momentum, np.copy(p_v) * P_MASS)

        steps_per_it = 10
        print(initial_momentum)
        for _ in range(20):
            self.system.integrator.run(steps_per_it)

            p_v = self.lbf.get_particle_velocity(P_UID)
            p_pos = self.lbf.get_particle_position(P_UID)

            fluid_mom = get_fluid_mom()
            particle_mom = p_v * P_MASS
            measured_momentum = fluid_mom + particle_mom

            np.testing.assert_allclose(measured_momentum,
                                        initial_momentum, atol=1E-5)


@utx.skipIfMissingFeatures(['LB_WALBERLA'])
class LBWalberlaMomentum(ut.TestCase, Momentum):

    def setUp(self):
        self.system.actors.clear()
        self.system.part.clear()


if __name__ == "__main__":
    ut.main()
