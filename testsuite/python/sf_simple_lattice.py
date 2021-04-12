#
# Copyright (C) 2017-2021 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published byss
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
#

import unittest as ut
import espressomd
import numpy as np
import itertools


class StructureFactorTest(ut.TestCase):
    '''
    Test structure factor analysis against rectangular lattices.
    We do not check the wavevectors directly, but rather the
    corresponding SF order, which is more readable (integer value).
    '''

    box_l = 16
    part_ty = 0
    sf_order = 16
    system = espressomd.System(box_l=[box_l, box_l, box_l])

    def tearDown(self):
        self.system.part.clear()

    def order(self, wavevectors, a):
        """
        Square and rescale wavevectors to recover the corresponding
        SF order, which is an integer between 0 and ``self.sf_order``.
        """
        return (wavevectors * a / (2 * np.pi))**2

    def generate_peaks(self, a, b, c, conditions):
        '''
        Generate the main diffraction peaks for crystal structures.

        Parameters
        ----------
        a: :obj:`float`
            Length of the unit cell on the x-axis.
        b: :obj:`float`
            Length of the unit cell on the y-axis.
        c: :obj:`float`
            Length of the unit cell on the z-axis.
        conditions: :obj:`function`
            Reflection conditions for the crystal lattice.
        '''
        hkl_ranges = [
            range(0, self.sf_order + 1),
            range(-self.sf_order, self.sf_order + 1),
            range(-self.sf_order, self.sf_order + 1),
        ]
        reflections = [np.linalg.norm([h / a, k / b, l / c]) * 2 * np.pi
                       for (h, k, l) in itertools.product(*hkl_ranges)
                       if conditions(h, k, l) and (h + k + l != 0) and
                       (h**2 + k**2 + l**2) <= self.sf_order**2]
        return np.unique(reflections)

    def test_tetragonal(self):
        """Check tetragonal lattice."""
        a = 2
        b = 4
        c = 8
        xen = range(0, self.box_l, a)
        yen = range(0, self.box_l, b)
        zen = range(0, self.box_l, c)
        for i, j, k in itertools.product(xen, yen, zen):
            self.system.part.add(type=self.part_ty, pos=(i, j, k))
        wavevectors, intensities = self.system.analysis.structure_factor(
            sf_types=[self.part_ty], sf_order=self.sf_order)
        intensities = np.around(intensities, 8)
        # no reflection conditions on (h,k,l)
        peaks_ref = self.generate_peaks(a, b, c, lambda h, k, l: True)
        peaks = wavevectors[np.nonzero(intensities)]
        np.testing.assert_array_almost_equal(
            self.order(peaks, a), self.order(peaks_ref[:len(peaks)], a))

    def test_sc(self):
        """Check simple cubic lattice."""
        l0 = 4
        xen = range(0, self.box_l, l0)
        for i, j, k in itertools.product(xen, repeat=3):
            self.system.part.add(type=self.part_ty, pos=(i, j, k))
        wavevectors, intensities = self.system.analysis.structure_factor(
            sf_types=[self.part_ty], sf_order=self.sf_order)
        intensities = np.around(intensities, 8)
        np.testing.assert_array_equal(
            intensities[np.nonzero(intensities)], len(self.system.part))
        # no reflection conditions on (h,k,l)
        peaks = wavevectors[np.nonzero(intensities)]
        peaks_ref = self.generate_peaks(l0, l0, l0, lambda h, k, l: True)
        np.testing.assert_array_almost_equal(
            self.order(peaks, l0), self.order(peaks_ref[:len(peaks)], l0))

    def test_bcc(self):
        """Check body-centered cubic lattice."""
        l0 = 4
        m = l0 / 2
        xen = range(0, self.box_l, l0)
        for i, j, k in itertools.product(xen, repeat=3):
            self.system.part.add(type=self.part_ty, pos=(i, j, k))
            self.system.part.add(type=self.part_ty, pos=(i + m, j + m, k + m))
        wavevectors, intensities = self.system.analysis.structure_factor(
            sf_types=[self.part_ty], sf_order=self.sf_order)
        intensities = np.around(intensities, 8)
        np.testing.assert_array_equal(
            intensities[np.nonzero(intensities)], len(self.system.part))
        # reflection conditions
        # (h+k+l) even => F = 2f, otherwise F = 0
        peaks_ref = self.generate_peaks(
            l0, l0, l0, lambda h, k, l: (h + k + l) % 2 == 0)
        peaks = wavevectors[np.nonzero(intensities)]
        np.testing.assert_array_almost_equal(
            self.order(peaks, l0), self.order(peaks_ref[:len(peaks)], l0))

    def test_fcc(self):
        """Check face-centered cubic lattice."""
        l0 = 4
        m = l0 / 2
        xen = range(0, self.box_l, l0)
        for i, j, k in itertools.product(xen, repeat=3):
            self.system.part.add(type=self.part_ty, pos=(i, j, k))
            self.system.part.add(type=self.part_ty, pos=(i + m, j + m, k))
            self.system.part.add(type=self.part_ty, pos=(i + m, j, k + m))
            self.system.part.add(type=self.part_ty, pos=(i, j + m, k + m))
        wavevectors, intensities = self.system.analysis.structure_factor(
            sf_types=[self.part_ty], sf_order=self.sf_order)
        intensities = np.around(intensities, 8)
        np.testing.assert_array_equal(
            intensities[np.nonzero(intensities)], len(self.system.part))
        # reflection conditions
        # (h,k,l) all even or odd => F = 4f, otherwise F = 0
        peaks_ref = self.generate_peaks(
            l0, l0, l0, lambda h, k, l:
            h % 2 == 0 and k % 2 == 0 and l % 2 == 0 or
            h % 2 == 1 and k % 2 == 1 and l % 2 == 1)
        peaks = wavevectors[np.nonzero(intensities)]
        np.testing.assert_array_almost_equal(
            self.order(peaks, l0), self.order(peaks_ref[:len(peaks)], l0))

    def test_cco(self):
        """Check c-centered orthorhombic lattice."""
        l0 = 4
        m = l0 / 2
        xen = range(0, self.box_l, l0)
        for i, j, k in itertools.product(xen, repeat=3):
            self.system.part.add(type=self.part_ty, pos=(i, j, k))
            self.system.part.add(type=self.part_ty, pos=(i + m, j + m, k))
        wavevectors, intensities = self.system.analysis.structure_factor(
            sf_types=[self.part_ty], sf_order=self.sf_order)
        intensities = np.around(intensities, 8)
        # reflection conditions
        # (h+k) even => F = 2f, otherwise F = 0
        peaks_ref = self.generate_peaks(
            l0, l0, l0, lambda h, k, l: (h + k) % 2 == 0)
        peaks = wavevectors[np.nonzero(intensities)]
        np.testing.assert_array_almost_equal(
            self.order(peaks, l0), self.order(peaks_ref[:len(peaks)], l0))

    def test_exceptions(self):
        with self.assertRaisesRegex(ValueError, 'order has to be a strictly positive number'):
            self.system.analysis.structure_factor(sf_types=[0], sf_order=0)


if __name__ == "__main__":
    ut.main()
