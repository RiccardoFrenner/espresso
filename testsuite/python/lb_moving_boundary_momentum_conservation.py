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
GRID_SIZE = 12
KVISC = 5e-3 * AGRID**2 / TIME_STEP
DENS = 1 #AGRID**(-3)
F = 5.5 / GRID_SIZE**3
# F = 0
GAMMA = 1

SYSTEM_VOLUME = (AGRID*GRID_SIZE)**3
# EXTERNAL_FORCE_DENSITY = [-.7 * F, .9 * F, .8 * F]
EXTERNAL_FORCE_DENSITY = [0., 0., 0.]
EXT_FLUID_FORCE = SYSTEM_VOLUME * np.array(EXTERNAL_FORCE_DENSITY)
EXT_PARTICLE_FORCE = [0., 0., -F]

PE_PARAMS = ([(EXT_PARTICLE_FORCE, "External particle force")],)
# PE_PARAMS = ([],)

LB_PARAMS = {'agrid': AGRID,
             'dens': DENS,
             'visc': KVISC,
             'tau': TIME_STEP,
             'ext_force_density': EXTERNAL_FORCE_DENSITY,
             "pe_params": PE_PARAMS}


class Momentum(object):
    """
    Tests momentum conservation for an LB coupled to an extended particle,
    where an external force is applied to the lb fluid but not to the particle.
    The total momentum change due to impulse is approximated.
    """
    lbf = None
    system = espressomd.System(box_l=[GRID_SIZE * AGRID] * 3)
    system.time_step = TIME_STEP
    system.cell_system.skin = 0.01

    def test(self):
        self.system.actors.clear()
        self.system.part.clear()
        self.system.actors.add(self.lbf)
        self.system.thermostat.set_lb(LB_fluid=self.lbf, gamma=GAMMA, seed=1)
        np.testing.assert_allclose(
            self.lbf.ext_force_density,
            LB_PARAMS["ext_force_density"])

        # Initial momentum before integration = 0
        np.testing.assert_allclose(
            self.system.analysis.linear_momentum(), [0., 0., 0.], atol=1E-12)

        p_uid = 0
        p_initial_v = np.array([0,0,0])
        p_radius = 2 * AGRID
        p_density = 1.1 * DENS

        self.lbf.create_particle_material("myMat", p_density)
        self.lbf.add_particle(p_uid, self.system.box_l/2, p_radius, p_initial_v)

        p_pos = self.lbf.get_particle_position(p_uid)
        p_v = self.lbf.get_particle_velocity(p_uid)

        # Check unit conversion
        np.testing.assert_allclose(np.copy(p_pos), np.copy(self.system.box_l/2))
        np.testing.assert_allclose(np.copy(p_v), p_initial_v)

        get_fluid_mom = lambda : np.array(
            self.system.analysis.linear_momentum(include_particles=False))

        p_mass = 4./3. * np.pi * np.power(p_radius,3) * p_density
        initial_momentum = get_fluid_mom() + p_initial_v * p_mass
        np.testing.assert_allclose(initial_momentum, np.copy(p_v) * p_mass)

        steps_per_it = 1
        external_impulse = 0
        print(initial_momentum)
        for i in count(0, 1):
            # print(f"Iteration: {i*steps_per_it}")
            self.system.integrator.run(steps_per_it)

            p_v = self.lbf.get_particle_velocity(p_uid)
            p_pos = self.lbf.get_particle_position(p_uid)

            # get_particle_force only returns hydrodynamic force
            p_f = self.lbf.get_particle_force(p_uid) + EXT_PARTICLE_FORCE

            measured_momentum = get_fluid_mom() + p_v * p_mass

            fluid_volume = SYSTEM_VOLUME - p_mass/p_density
            # compensation due to external impulse (dp = dt * F)
            external_impulse += TIME_STEP * (
                np.array(EXTERNAL_FORCE_DENSITY) * fluid_volume # Force on fluid
                + EXT_PARTICLE_FORCE) # Force on particle

            with np.printoptions(precision=2, floatmode='fixed'):
                fm = measured_momentum - p_v * p_mass
                pm = p_v * p_mass
                print("| {:5.2e} | {:5.2e} | {:5.2e} | {:5.2e} |".format(
                    external_impulse[2], measured_momentum[2], fm[2], pm[2]))

            # np.testing.assert_allclose(measured_momentum,
            #                            external_impulse, atol=1E-2)

            if np.any(np.abs(p_pos - .5*self.system.box_l) > .5*self.system.box_l - 1.5*p_radius):
                print(f"Particle ({p_pos}) too close to boundary ({self.system.box_l}).\nStopping simulation...")
                break


@utx.skipIfMissingFeatures(['LB_WALBERLA'])
class LBWalberlaMomentum(ut.TestCase, Momentum):

    def setUp(self):
        self.lbf = espressomd.lb.LBFluidWalberla(**LB_PARAMS)


if __name__ == "__main__":
    ut.main()
