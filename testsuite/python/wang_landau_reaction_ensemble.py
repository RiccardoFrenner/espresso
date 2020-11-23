#
# Copyright (C) 2013-2019 The ESPResSo project
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
#
"""Testmodule for the Wang-Landau Reaction Ensemble.
"""
import numpy as np
import unittest as ut

import espressomd
from espressomd.interactions import HarmonicBond
from espressomd import reaction_ensemble


class ReactionEnsembleTest(ut.TestCase):

    """Test the core implementation of the wang_landau reaction ensemble.

    Create a harmonic bond between the two reacting particles. Therefore the
    potential energy is quadratic in the elongation of the bond and
    therefore the density of states is known as the one of the harmonic
    oscillator
    """

    # System parameters
    #
    box_l = 6 * np.sqrt(2)
    temperature = 1.0

    # Integration parameters
    #
    system = espressomd.System(box_l=[box_l, box_l, box_l])
    np.random.seed(seed=42)
    system.time_step = 0.01
    system.cell_system.skin = 0
    system.cell_system.set_n_square(use_verlet_lists=False)

    #
    # Setup System
    #

    N0 = 1  # number of titratable units
    K_diss = 0.0088

    system.part.add(id=0, pos=[0, 0, 0] * system.box_l, type=3)
    system.part.add(id=1, pos=[1.0, 1.0, 1.0] * system.box_l / 2.0, type=1)
    system.part.add(id=2, pos=np.random.random() * system.box_l, type=2)
    system.part.add(id=3, pos=np.random.random() * system.box_l, type=2)

    h = HarmonicBond(r_0=0, k=1)
    system.bonded_inter[0] = h
    system.part[0].add_bond((h, 1))
    WLRE = reaction_ensemble.WangLandauReactionEnsemble(
        temperature=temperature, exclusion_radius=0, seed=86)
    WLRE.add_reaction(
        gamma=K_diss, reactant_types=[0], reactant_coefficients=[1],
        product_types=[1, 2], product_coefficients=[1, 1],
        default_charges={0: 0, 1: -1, 2: +1})
    system.setup_type_map([0, 1, 2, 3])
    # initialize wang_landau
    file_input = "energy_boundaries.dat"
    file_output = "WL_potential_out.dat"
    # generate preliminary_energy_run_results here, this should be done in a
    # separate simulation without energy reweighting using the update energy
    # functions
    np.savetxt(file_input, np.transpose([[0, 1], [0, 0], [9, 9]]),
               delimiter='\t', header="nbar   E_potmin   E_potmax")

    WLRE.add_collective_variable_degree_of_association(
        associated_type=0, min=0, max=1, corresponding_acid_types=[0, 1])
    WLRE.set_wang_landau_parameters(
        final_wang_landau_parameter=0.8 * 1e-2,
        do_not_sample_reaction_partition_function=True,
        full_path_to_output_filename=file_output)

    def test_wang_landau_energy_recording(self):
        self.WLRE.update_maximum_and_minimum_energies_at_current_state()
        self.WLRE.write_out_preliminary_energy_run_results()
        nbars, E_mins, E_maxs = np.loadtxt(
            "preliminary_energy_run_results", unpack=True)
        np.testing.assert_almost_equal(nbars, [0, 1])
        np.testing.assert_almost_equal(E_mins, [27.0, -10])
        np.testing.assert_almost_equal(E_maxs, [27.0, -10])

    def test_wang_landau_output(self):
        self.WLRE.add_collective_variable_potential_energy(
            filename=self.file_input, delta=0.05)

        # run MC until convergence
        while True:
            try:
                self.WLRE.reaction()
                for _ in range(2):
                    self.WLRE.displacement_mc_move_for_particles_of_type(3)
            except reaction_ensemble.WangLandauHasConverged:
                break

        nbars, Epots, WL_potentials = np.loadtxt(self.file_output, unpack=True)
        mask_nbar_0 = np.where(np.abs(nbars - 1.0) < 0.0001)
        Epots = Epots[mask_nbar_0][1:]
        WL_potentials = WL_potentials[mask_nbar_0][1:]

        def calc_from_partition_function(quantity):
            probability = np.exp(WL_potentials - Epots / self.temperature)
            return np.sum(quantity * probability) / np.sum(probability)

        # calculate the canonical potential energy
        pot_energy = calc_from_partition_function(Epots)
        # calculate the canonical configurational heat capacity
        pot_energy_sq = calc_from_partition_function(Epots**2)
        heat_capacity = pot_energy_sq - pot_energy**2

        # for the calculation regarding the analytical results which are
        # compared here, see Master Thesis Jonas Landsgesell p. 72
        self.assertAlmostEqual(
            pot_energy, 1.5, places=1,
            msg="difference to analytical expected canonical potential energy too big")
        self.assertAlmostEqual(
            heat_capacity, 1.5, places=1,
            msg="difference to analytical expected canonical configurational heat capacity too big")

    def _wang_landau_output_checkpoint(self, filename):
        # write first checkpoint
        self.WLRE.write_wang_landau_checkpoint()
        old_checkpoint = np.loadtxt(filename)

        # modify old_checkpoint in memory and in file (this destroys the
        # information contained in the checkpoint, but allows for testing of
        # the functions)
        modified_checkpoint = old_checkpoint
        modified_checkpoint[0] = 1
        np.savetxt(filename, modified_checkpoint)

        # check whether changes are carried out correctly
        self.WLRE.load_wang_landau_checkpoint()
        self.WLRE.write_wang_landau_checkpoint()
        new_checkpoint = np.loadtxt(filename)
        np.testing.assert_almost_equal(new_checkpoint, modified_checkpoint)

    def test_wang_landau_output_checkpoint(self):
        filenames = ["checkpoint_wang_landau_potential_checkpoint",
                     "checkpoint_wang_landau_histogram_checkpoint"]
        for filename in filenames:
            self._wang_landau_output_checkpoint(filename)


if __name__ == "__main__":
    ut.main()
