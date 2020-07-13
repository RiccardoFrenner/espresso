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
import unittest as ut
import unittest_decorators as utx
import numpy as np

import espressomd.lb
from tests_common import single_component_maxwell

"""
Check the lattice-Boltzmann thermostat with respect to the particle velocity
distribution.


"""

KT = 1 
AGRID = 1
VISC = 4 
DENS = 1.7
TIME_STEP = 0.05
LB_PARAMS = {'agrid': AGRID,
             'dens': DENS,
             'visc': VISC,
             'tau': TIME_STEP,
             'kT': 1 * KT,
             'seed': 123}


class LBThermostatCommon:

    """Base class of the test that holds the test logic."""
    lbf = None
    system = espressomd.System(box_l=[10.0, 10.0, 10.0])
    system.time_step = TIME_STEP
    system.cell_system.skin = 0.4 * AGRID

    def test_fluid(self):
        self.prepare()
        self.system.integrator.run(100)
        vs = []
        for _ in range(1000):
            v = self.lbf[0, 0, 0].velocity
            vs.append(v)
            print(v)
            self.system.integrator.run(1)

        print(np.var(v), np.mean(v))
        self.assertAlmostEqual(np.var(v), KT, delta=1E-3)

    def prepare(self):
        self.system.actors.clear()
        self.system.actors.add(self.lbf)
        self.system.thermostat.set_lb(LB_fluid=self.lbf, seed=5, gamma=5.0)

    def ixxtest_with_particles(self):
        self.prepare()
        self.system.part.add(
            pos=np.random.random((100, 3)) * self.system.box_l)
        self.system.integrator.run(20)
        N = len(self.system.part)
        loops = 250
        v_stored = np.zeros((N * loops, 3))
        for i in range(loops):
            self.system.integrator.run(3)
            v_stored[i * N:(i + 1) * N, :] = self.system.part[:].v
        minmax = 3
        n_bins = 7
        for i in range(3):
            hist = np.histogram(v_stored[:, i], range=(-minmax, minmax),
                                bins=n_bins, density=False)
            data = hist[0] / float(v_stored.shape[0])
            bins = hist[1]
            expected = [single_component_maxwell(
                bins[j], bins[j + 1], KT) for j in range(n_bins)]
            print(data)
            print(expected)
            print()


@utx.skipIfMissingFeatures("LB_WALBERLA")
class LBWalberlaThermostat(ut.TestCase, LBThermostatCommon):

    """Test for the CPU implementation of the LB."""

    def setUp(self):
        self.lbf = espressomd.lb.LBFluidWalberla(**LB_PARAMS)


if __name__ == '__main__':
    ut.main()
