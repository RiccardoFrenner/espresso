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
TIME_STEP = 0.008
AGRID = .4
GRID_SIZE = 12
KVISC = 5e-3 * AGRID**2 / TIME_STEP
DENS = AGRID**(-3)
F = 5.5 / GRID_SIZE**3
# F = 0
GAMMA = 1

SYSTEM_VOLUME = (AGRID*GRID_SIZE)**3
EXTERNAL_FORCE_DENSITY = [-.7 * F, .9 * F, .8 * F]
EXT_FLUID_FORCE = SYSTEM_VOLUME * np.array(EXTERNAL_FORCE_DENSITY)

# PE_PARAMS = ([(-EXT_FLUID_FORCE, "Minus external fluid force")],)
PE_PARAMS = ([],)

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
        p_v_lu = np.array([0, 0, 0])
        p_radius_lu = 2
        p_density_lu = 1.1

        p_radius = p_radius_lu * AGRID
        p_density = p_density_lu * AGRID**(-3)

        p_mass = 4./3. * np.pi * np.power(p_radius,3) * p_density
        self.lbf.create_particle_material("myMat", p_density_lu)

        self.lbf.add_particle(p_uid, [GRID_SIZE//2]*3, p_radius_lu, p_v_lu)
        # p = self.system.part.add(pos=self.system.box_l / 2, virtual=True)

        p_pos = self.lbf.get_particle_position(p_uid)
        p_v = self.lbf.get_particle_velocity(p_uid)

        # Check unit conversion
        np.testing.assert_allclose(np.copy(p_pos), np.copy(self.system.box_l/2))
        np.testing.assert_allclose(np.copy(p_v), AGRID/TIME_STEP*p_v_lu)

        initial_momentum = np.array(self.system.analysis.linear_momentum(include_particles=False)) + p_v * p_mass
        np.testing.assert_allclose(initial_momentum, np.copy(p_v) * p_mass)
        steps_per_it = 1
        compensation = 0
        print(initial_momentum)
        for i in count(0, 1):
            # print(f"Iteration: {i*steps_per_it}")
            self.system.integrator.run(steps_per_it)

            p_v = self.lbf.get_particle_velocity(p_uid)
            p_pos = self.lbf.get_particle_position(p_uid)
            p_f = self.lbf.get_particle_force(p_uid)# - EXT_FLUID_FORCE
            measured_momentum = self.system.analysis.linear_momentum(include_particles=False) + p_v * p_mass
            coupling_force = -p_f #- EXT_FLUID_FORCE
            # compensation = -TIME_STEP / 2 * coupling_force
            compensation += TIME_STEP * np.array(EXTERNAL_FORCE_DENSITY) * (SYSTEM_VOLUME - p_mass/p_density)

            with np.printoptions(precision=3, floatmode='fixed'):
                print(measured_momentum, measured_momentum - p_v * p_mass, p_v * p_mass)
            np.testing.assert_allclose(measured_momentum,
                                       compensation, atol=1E-2)
            if np.any(np.abs(p_pos - .5*self.system.box_l) > .5*self.system.box_l - 1.5*p_radius):
                print(f"Particle ({p_pos}) too close to boundary ({self.system.box_l}).\nStopping simulation...")
                break
            # if np.linalg.norm(p_f) < 0.01 \
            #    and np.all(np.abs(p_pos) > 10.1 * self.system.box_l):
            #     break

        # # Make sure, the particle has crossed the periodic boundaries
        # self.assertGreater(
        #     max(
        #         np.abs(p.v) *
        #         self.system.time),
        #     self.system.box_l[0])


@utx.skipIfMissingFeatures(['LB_WALBERLA'])
class LBWalberlaMomentum(ut.TestCase, Momentum):

    def setUp(self):
        self.lbf = espressomd.lb.LBFluidWalberla(**LB_PARAMS)


if __name__ == "__main__":
    ut.main()
