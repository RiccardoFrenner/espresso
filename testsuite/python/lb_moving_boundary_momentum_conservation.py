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

try:
    from espressomd.virtual_sites import VirtualSitesWalberlaMovingBoundary
except ImportError:
    pass

import unittest as ut

import unittest_decorators as utx

import numpy as np

# Define the LB Parameters
TIME_STEP = 0.008
AGRID = .4
GRID_SIZE = 6
KVISC = 4
DENS = 2.3
F = 5.5 / GRID_SIZE**3
GAMMA = 1

UID = 42
PARTICLE_RADIUS = GRID_SIZE * AGRID / 100
PARTICLE_VOLUME = 4.0*np.pi/3.0*PARTICLE_RADIUS**3
PARTICLE_DENSITY = 1. / PARTICLE_VOLUME
PARTICLE_MASS = PARTICLE_DENSITY*PARTICLE_VOLUME
PARTICLE_VELOCITY = np.array([5.5 / GRID_SIZE**3] * 3)

# CONST_GLOBAL_FORCES = [
#     ([0,0,0], "Gravitational Force"),
#     ([0,0,0], "No Force")
# ]

CONST_GLOBAL_FORCES = []

PE_PARAMS = (
    True,
    True,
    1.5,
    True,
    1,
    CONST_GLOBAL_FORCES
)

LB_PARAMS = {'agrid': AGRID,
             'dens': DENS,
             'visc': KVISC,
             'tau': TIME_STEP,
             'ext_force_density': [-.7 * F, .9 * F, .8 * F],
             'pe_params': PE_PARAMS}


class Momentum(object):
    """
    Tests momentum conservation for an LB coupled to a particle, where opposing
    forces are applied to LB and particle. The test should uncover issues
    with boundary and ghost layer handling.

    """
    lbf = None
    system = espressomd.System(box_l=[GRID_SIZE * AGRID] * 3)
    system.time_step = TIME_STEP
    system.cell_system.skin = 0.01

    SYSTEM_VOLUME = (GRID_SIZE * AGRID)**3

    # def test_moment_conservation(self):
    #     CONST_GLOBAL_FORCES = []
    #     PE_PARAMS = (
    #         True,
    #         True,
    #         1.5,
    #         True,
    #         1,
    #         CONST_GLOBAL_FORCES
    #     )
    #     LB_PARAMS = {'agrid': AGRID,
    #                 'dens': DENS,
    #                 'visc': KVISC,
    #                 'tau': TIME_STEP,
    #                 'pe_params': PE_PARAMS}

    #     self.lbf = espressomd.lb.LBFluidWalberla(**LB_PARAMS)

    #     self.system.actors.clear()
    #     self.system.part.clear()
    #     self.system.actors.add(self.lbf)
    #     self.system.thermostat.set_lb(LB_fluid=self.lbf, gamma=GAMMA, seed=1)

    #     # Initial momentum before integration = 0
    #     np.testing.assert_allclose(
    #         self.system.analysis.linear_momentum(), [0., 0., 0.], atol=1E-12)

    #     particle_pos = self.system.box_l / 2
    #     particle_momentum = PARTICLE_MASS * PARTICLE_VELOCITY

    #     self.lbf.create_particle_material("myMat", PARTICLE_DENSITY)
    #     self.lbf.add_particle(UID, particle_pos, PARTICLE_RADIUS, vel=PARTICLE_VELOCITY, material_name="myMat")


    #     print(PARTICLE_MASS * PARTICLE_VELOCITY)

    #     self.system.integrator.run(100)
    #     particle_pos = self.lbf.get_particle_position(UID)
    #     particle_momentum = self.lbf.get_particle_velocity(UID) * PARTICLE_MASS

    #     print(particle_momentum)
    #     print(np.array(self.system.analysis.linear_momentum()))
    #     # np.testing.assert_allclose(PARTICLE_MASS * PARTICLE_VELOCITY, np.array(self.system.analysis.linear_momentum()) + particle_momentum)

    def test_external_force(self):
        EXT_FORCE_DENSITY = [-.7 * F, .9 * F, .8 * F]
        CONST_GLOBAL_FORCES = [(-np.array(EXT_FORCE_DENSITY)*Momentum.SYSTEM_VOLUME, "negative_ext_fluid_force")]
        PE_PARAMS = (
            True,
            True,
            1.5,
            True,
            1,
            CONST_GLOBAL_FORCES
        )
        LB_PARAMS = {'agrid': AGRID,
                    'dens': DENS,
                    'visc': KVISC,
                    'tau': TIME_STEP,
                    'ext_force_density': EXT_FORCE_DENSITY,
                    'pe_params': PE_PARAMS}

        self.lbf = espressomd.lb.LBFluidWalberla(**LB_PARAMS)

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

        self.lbf.create_particle_material("myMat", PARTICLE_DENSITY)
        self.lbf.add_particle(UID, self.system.box_l / 2, PARTICLE_RADIUS, vel=[.2, .4, .6], material_name="myMat")

        initial_momentum = self.lbf.get_particle_velocity(UID) * PARTICLE_MASS

        with np.printoptions(precision=8, suppress=True, floatmode='fixed'):
            print(np.array(CONST_GLOBAL_FORCES[0][0]))
        while True:
            pf = self.lbf.get_particle_force(UID)
            pv = self.lbf.get_particle_velocity(UID)
            ppos = self.lbf.get_particle_position(UID)
            with np.printoptions(precision=4, suppress=True, floatmode='fixed'):
                print(pf, pv, ppos)

            self.system.integrator.run(500)

            measured_momentum = self.system.analysis.linear_momentum() + self.lbf.get_particle_velocity(UID) * PARTICLE_MASS
            coupling_force = -(pf - CONST_GLOBAL_FORCES[0][0])
            compensation = -TIME_STEP / 2 * coupling_force

            np.testing.assert_allclose(measured_momentum + compensation,
                                       initial_momentum, atol=1E-4)
            if np.linalg.norm(pf) < 0.01 \
               and np.all(np.abs(ppos) > 10.1 * self.system.box_l):
                break

        # Make sure, the particle has crossed the periodic boundaries
        self.assertGreater(
            max(
                np.abs(p.v) *
                self.system.time),
            self.system.box_l[0])


@utx.skipIfMissingFeatures(['LB_WALBERLA'])
class LBWalberlaMomentum(ut.TestCase, Momentum):

    def setUp(self):
        pass


if __name__ == "__main__":
    ut.main()
