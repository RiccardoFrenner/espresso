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
if espressomd.has_features("VIRTUAL_SITES"):
    from espressomd.virtual_sites import VirtualSitesWalberlaMovingBoundary
import unittest as ut

import unittest_decorators as utx

import numpy as np
from itertools import count

# Define the LB Parameters
TIME_STEP = 1 #0.008
AGRID = 1#.4
GRID_SIZE = 54
KVISC = 0.1 * AGRID**2 / TIME_STEP
DENS = 1 * AGRID**(-3)
P_UID_1 = 0
P_UID_2 = 1
P_DENSITY = 1.1 * DENS
P_RADIUS = 7.5 * AGRID

P_VOLUME = 4./3. * np.pi * np.power(P_RADIUS,3)
P_MASS = P_VOLUME * P_DENSITY

GAMMA = 0

LB_PARAMS = {
    'agrid': AGRID,
        'dens': DENS,
        'visc': KVISC,
        'tau': TIME_STEP,
        "pe_params": ([], )
}

@utx.skipIfMissingFeatures(['LB_WALBERLA', 'EXTERNAL_FORCES'])
class MBVirtualSitesForces(ut.TestCase):
    """
    Tests the VirtualSitesWalberlaMovingBoundary implementation.
    """
    lbf = None
    system = espressomd.System(box_l=[GRID_SIZE * AGRID] * 3)
    system.time_step = TIME_STEP
    system.cell_system.skin = 0.01

    # Needs to be at least this big to be equal to walberlas ghost particle setup
    system.min_global_cut = P_RADIUS + system.cell_system.skin

    def tearDown(self):
        self.system.part.clear()
        self.system.actors.clear()
        self.system.thermostat.turn_off()

    def setUp(self):
        self.system.part.clear()
        self.system.actors.clear()
        self.system.virtual_sites = VirtualSitesWalberlaMovingBoundary()

        self.lbf = espressomd.lb.LBFluidWalberla(**LB_PARAMS)
        self.system.actors.add(self.lbf)
        self.system.thermostat.set_lb(
            LB_fluid=self.lbf,
            act_on_virtual=False,
            gamma=GAMMA, seed=1)

        self.assertIsInstance(self.system.virtual_sites, VirtualSitesWalberlaMovingBoundary)

        # Add pe particles
        p_initial_v = np.array([0,0,0])
        self.lbf.create_particle_material("myMat", P_DENSITY)
        self.lbf.add_particle(P_UID_1, self.system.box_l/2 - np.array([2*P_RADIUS,0,0]), P_RADIUS, p_initial_v, "myMat")
        self.lbf.add_particle(P_UID_2, self.system.box_l/2 + np.array([2*P_RADIUS,0,0]), P_RADIUS, p_initial_v, "myMat")
        self.lbf.finish_particle_adding()

        # Add VS
        self.system.part.add(pos=self.lbf.get_particle_position(P_UID_1), virtual=True, id=P_UID_1)
        self.system.part.add(pos=self.lbf.get_particle_position(P_UID_2), virtual=True, id=P_UID_2)

    # def test_VS_force_single_particle(self):
    #     """
    #     Tests the VS back transfer of forces and torques method.
    #     """

    #     p = self.system.part[P_UID_1]
    #     self.assertIsInstance(p, espressomd.particle_data.ParticleHandle)

    #     np.testing.assert_equal(self.lbf.get_particle_velocity(P_UID_1), np.array([0,0,0]))
    #     np.testing.assert_equal(self.lbf.get_particle_position(P_UID_1), self.system.box_l/2 - np.array([2*P_RADIUS,0,0]))

    #     p_pos = self.lbf.get_particle_position(P_UID_1)
    #     p_v = self.lbf.get_particle_velocity(P_UID_1)


    #     F = np.array([0,0,1e-1])
    #     p.ext_force = np.copy(F)

    #     print("timestep\tpx\tpy\tpz\t|pv|")
    #     timesteps = 40
    #     for i in range(timesteps):
    #         print(i, p_pos[0], p_pos[1], p_pos[2], np.linalg.norm(np.copy(p_v)), sep="\t")

    #         np.testing.assert_equal(np.array(p.ext_force), np.copy(F))
    #         self.system.integrator.run(1)
    #         np.testing.assert_equal(np.array(p.ext_force), np.copy(F))

    #         p_pos = self.lbf.get_particle_position(P_UID_1)
    #         p_v = self.lbf.get_particle_velocity(P_UID_1)

    #         # Position and velocity of VS and pe particle should be equal
    #         np.testing.assert_equal(np.copy(p_v), np.copy(p.v))
    #         np.testing.assert_equal(np.copy(p_pos), np.copy(p.pos_folded))

    #     # pe particle should a velocity from the force
    #     appr_exp_p_vel = F / P_MASS * timesteps * TIME_STEP
    #     self.assertGreater(np.linalg.norm(np.copy(p_v)), 1e-2*np.linalg.norm(appr_exp_p_vel))

    #     self.assertEqual(self.system.min_global_cut, P_RADIUS + self.system.cell_system.skin)

    def test_VS_force(self):
        """
        Tests the VS back transfer of forces and torques method.
        """


        p1 = self.system.part[P_UID_1]
        p2 = self.system.part[P_UID_2]

        p1.ext_force = np.array([0,0,0])

        r_0 = 5*P_RADIUS
        bond = espressomd.interactions.HarmonicBond(k=1, r_0=r_0)
        self.system.bonded_inter.add(bond)
        p1.add_bond((bond, p2))

        self.assertIsInstance(p1, espressomd.particle_data.ParticleHandle)
        self.assertIsInstance(p2, espressomd.particle_data.ParticleHandle)

        np.testing.assert_equal(self.lbf.get_particle_velocity(P_UID_1), np.array([0,0,0]))
        np.testing.assert_equal(self.lbf.get_particle_velocity(P_UID_2), np.array([0,0,0]))

        np.testing.assert_equal(self.lbf.get_particle_position(P_UID_1), self.system.box_l/2 - np.array([2*P_RADIUS,0,0]))
        np.testing.assert_equal(self.lbf.get_particle_position(P_UID_2), self.system.box_l/2 + np.array([2*P_RADIUS,0,0]))

        # Check that particles are less than r_0 away from each other
        distance_old = np.linalg.norm(self.lbf.get_particle_position(P_UID_1) - self.lbf.get_particle_position(P_UID_1))
        self.assertLess(distance_old, r_0)

        # Therefore particles should repel eachother
        print("timestep\tdist")
        distance = distance_old
        i = 0
        while distance < r_0:
            print(i, distance, sep="\t")

            self.system.integrator.run(1)

            p1_pos = self.lbf.get_particle_position(P_UID_1)
            p2_pos = self.lbf.get_particle_position(P_UID_2)
            distance_old, distance = distance, np.linalg.norm(p1_pos - p2_pos)

            self.assertGreater(distance, distance_old)

            i+=1


        self.assertEqual(self.system.min_global_cut, P_RADIUS + self.system.cell_system.skin)


if __name__ == "__main__":
    ut.main()
