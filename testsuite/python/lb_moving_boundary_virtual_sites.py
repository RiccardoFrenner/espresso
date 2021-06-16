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
AGRID = 1 #.4
GRID_SIZE = 54
KVISC = 0.1 * AGRID**2 / TIME_STEP
DENS = 1 * AGRID**(-3)
P_UID = 12
P_DENSITY = 1.1 * DENS
P_RADIUS = 7.5 * AGRID

GAMMA = 1

PARTICLE_FORCE = np.array([2, -3, 5])*5e-2
LB_PARAMS = {
    'agrid': AGRID,
        'dens': DENS,
        'visc': KVISC,
        'tau': TIME_STEP,
        "pe_params": ([(PARTICLE_FORCE, "p_force")], )
}

@utx.skipIfMissingFeatures(['LB_WALBERLA'])
class MBVirtualSites(ut.TestCase):
    """
    Tests the VirtualSitesWalberlaMovingBoundary implementation.
    """
    lbf = None
    system = espressomd.System(box_l=[GRID_SIZE * AGRID] * 3)
    system.time_step = TIME_STEP
    system.cell_system.skin = 0.01

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

        # Add pe particle
        p_init_pos = self.system.box_l/2 + np.array([0,0, 0.5*self.system.box_l[2] - 1.2*P_RADIUS])
        p_initial_v = 0.1*np.copy(PARTICLE_FORCE)*AGRID
        self.lbf.create_particle_material("myMat", P_DENSITY)
        self.lbf.add_particle(P_UID, p_init_pos, P_RADIUS, p_initial_v, "myMat")
        self.lbf.finish_particle_adding()

        p_pos = self.lbf.get_particle_position(P_UID)

        # Add VS
        self.system.part.add(pos=p_pos, virtual=True, id=P_UID)

    def test_follow_pe_particle(self):
        """
        Tests whether the VS follows the pe particle in particular over the box
        and domain boundaries.
        Since periodic boundaries are enabled by default, passing the domain
        boundary should also not be a problem.
        """

        p = self.system.part[P_UID]

        p_pos = self.lbf.get_particle_position(P_UID)
        p_v = self.lbf.get_particle_velocity(P_UID)

        dist_travelled = 0
        steps_per_it = 20
        print("timestep\tpx\tpy\tpz\t|pv|")
        for i in count(0,1):
            print(i*steps_per_it, p_pos[0], p_pos[1], p_pos[2], np.linalg.norm(np.copy(p_v)), sep="\t")
            self.system.integrator.run(steps_per_it)

            p_pos = self.lbf.get_particle_position(P_UID)
            p_v = self.lbf.get_particle_velocity(P_UID)

            dist_travelled += np.linalg.norm(p_v)*TIME_STEP*steps_per_it

            # Position and velocity of VS and pe particle should be equal
            np.testing.assert_equal(np.copy(p_v), np.copy(p.v))
            np.testing.assert_equal(np.copy(p_pos), np.copy(p.pos_folded))


            # Stop after the particle has crossed the domain boundary
            if (dist_travelled > np.linalg.norm(self.system.box_l)): break
            if (i > 3): break


if __name__ == "__main__":
    ut.main()
