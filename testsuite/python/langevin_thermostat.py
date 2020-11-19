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
import unittest as ut
import unittest_decorators as utx
import espressomd
import numpy as np
from espressomd.accumulators import Correlator
from espressomd.observables import ParticleVelocities, ParticleBodyAngularVelocities
from thermostats_common import ThermostatsCommon


class ThermalizedBond(ut.TestCase, ThermostatsCommon):

    """Tests velocity distributions and diffusion for Langevin Dynamics"""
    system = espressomd.System(box_l=[1.0, 1.0, 1.0])
    system.cell_system.set_domain_decomposition(use_verlet_lists=True)
    system.cell_system.skin = 0
    system.periodicity = [0, 0, 0]
    system.integrator.set_vv()

    def setUp(self):
        np.random.seed(42)

    def tearDown(self):
        self.system.time_step = 1e-12
        self.system.cell_system.skin = 0.0
        self.system.part.clear()
        self.system.auto_update_accumulators.clear()

    def test_01__langevin_seed(self):
        """Test for RNG seed consistency."""
        system = self.system
        system.time_step = 0.01
        system.part.add(pos=[0, 0, 0])

        kT = 1.1
        gamma = 3.5

        # No seed should throw exception
        with self.assertRaises(ValueError):
            system.thermostat.set_langevin(kT=kT, gamma=gamma)

        system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=41)

        #'integrate 0' does not increase the philox counter and should give the same force
        system.integrator.run(0)
        force0 = np.copy(system.part[0].f)
        system.integrator.run(0)
        force1 = np.copy(system.part[0].f)
        np.testing.assert_almost_equal(force0, force1)

        #'integrate 1' should give a different force
        system.part.clear()
        system.part.add(pos=[0, 0, 0])
        system.integrator.run(1)
        force2 = np.copy(system.part[0].f)
        np.testing.assert_equal(np.all(np.not_equal(force1, force2)), True)

        # Different seed should give a different force
        system.part.clear()
        system.part.add(pos=[0, 0, 0])
        system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)
        system.integrator.run(1)
        force3 = np.copy(system.part[0].f)
        np.testing.assert_equal(np.all(np.not_equal(force2, force3)), True)

        # Same seed should not give the same force with different counter state
        system.part.clear()
        system.part.add(pos=[0, 0, 0])
        system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)
        system.integrator.run(1)
        force4 = np.copy(system.part[0].f)
        np.testing.assert_equal(np.all(np.not_equal(force3, force4)), True)

        # Seed offset should not give the same force with a lag
        system.part.clear()
        system.part.add(pos=[0, 0, 0])
        system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=41)
        system.integrator.run(1)
        system.part[0].pos = system.part[0].v = [0, 0, 0]
        system.integrator.run(1)
        force5 = np.copy(system.part[0].f)
        np.testing.assert_equal(np.all(np.not_equal(force4, force5)), True)

        # Same seed should give the same force with same counter state
        system.part.clear()
        system.part.add(pos=[0, 0, 0])
        system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)
        system.integrator.run(0, recalc_forces=True)
        force6 = np.copy(system.part[0].f)
        system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=42)
        system.integrator.run(0, recalc_forces=True)
        force7 = np.copy(system.part[0].f)
        np.testing.assert_almost_equal(force6, force7)

    def test_02__friction_trans(self):
        """Tests the translational friction-only part of the thermostat."""

        system = self.system
        # Translation
        gamma_t_i = 2
        gamma_t_a = np.array((0.5, 2, 1.5))
        v0 = np.array((5., 5., 5.))

        system.time_step = 0.0005
        system.part.add(pos=(0, 0, 0), v=v0)
        if espressomd.has_features("MASS"):
            system.part[0].mass = 3
        if espressomd.has_features("PARTICLE_ANISOTROPY"):
            system.thermostat.set_langevin(kT=0, gamma=gamma_t_a, seed=41)
        else:
            system.thermostat.set_langevin(kT=0, gamma=gamma_t_i, seed=41)

        system.time = 0
        for _ in range(100):
            system.integrator.run(10)
            if espressomd.has_features("PARTICLE_ANISOTROPY"):
                np.testing.assert_allclose(
                    np.copy(system.part[0].v),
                    v0 * np.exp(-gamma_t_a /
                                system.part[0].mass * system.time),
                    atol=4E-4)
            else:
                np.testing.assert_allclose(
                    np.copy(system.part[0].v),
                    v0 * np.exp(-gamma_t_i /
                                system.part[0].mass * system.time),
                    atol=45E-4)

    @utx.skipIfMissingFeatures("ROTATION")
    def test_03__friction_rot(self):
        """Tests the rotational friction-only part of the thermostat."""

        system = self.system
        # Translation
        gamma_t_i = 2
        gamma_t_a = [0.5, 2, 1.5]
        gamma_r_i = 3
        gamma_r_a = np.array((1.5, 0.7, 1.2))
        o0 = np.array((5., 5., 5.))

        system.time_step = 0.0001
        system.part.add(pos=(0, 0, 0), omega_body=o0, rotation=(1, 1, 1))
        if espressomd.has_features("ROTATIONAL_INERTIA"):
            system.part[0].rinertia = [2, 2, 2]
        if espressomd.has_features("PARTICLE_ANISOTROPY"):
            system.thermostat.set_langevin(
                kT=0, gamma=gamma_t_a, gamma_rotation=gamma_r_a, seed=41)
        else:
            system.thermostat.set_langevin(
                kT=0, gamma=gamma_t_i, gamma_rotation=gamma_r_i, seed=41)

        system.time = 0
        if espressomd.has_features("ROTATIONAL_INERTIA"):
            rinertia = np.copy(system.part[0].rinertia)
        else:
            rinertia = np.array((1, 1, 1))
        for _ in range(100):
            system.integrator.run(10)
            if espressomd.has_features("PARTICLE_ANISOTROPY"):
                np.testing.assert_allclose(
                    np.copy(system.part[0].omega_body),
                    o0 * np.exp(-gamma_r_a / rinertia * system.time), atol=5E-4)
            else:
                np.testing.assert_allclose(
                    np.copy(system.part[0].omega_body),
                    o0 * np.exp(-gamma_r_i / rinertia * system.time), atol=5E-4)

    def check_vel_dist_global_temp(self, recalc_forces, loops):
        """Test velocity distribution for global Langevin parameters.

        Parameters
        ----------
        recalc_forces : :obj:`bool`
            True if the forces should be recalculated after every step.
        loops : :obj:`int`
            Number of sampling loops
        """
        N = 200
        system = self.system
        system.time_step = 0.06
        kT = 1.1
        gamma = 3.5
        system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=41)
        v_minmax = 5
        bins = 4
        error_tol = 0.016
        self.check_global(
            N, kT, loops, v_minmax, bins, error_tol, recalc_forces)

    def test_vel_dist_global_temp(self):
        """Test velocity distribution for global Langevin parameters."""
        self.check_vel_dist_global_temp(False, loops=150)

    def test_vel_dist_global_temp_initial_forces(self):
        """Test velocity distribution for global Langevin parameters,
           when using the initial force calculation.
        """
        self.check_vel_dist_global_temp(True, loops=170)

    @utx.skipIfMissingFeatures("LANGEVIN_PER_PARTICLE")
    def test_vel_dist_per_particle(self):
        """Test Langevin dynamics with particle-specific kT and gamma. Covers
           all combinations of particle-specific gamma and temp set or not set.
        """
        N = 400
        system = self.system
        system.time_step = 0.06
        kT = 0.9
        gamma = 3.2
        gamma2 = 4.3
        kT2 = 1.5
        system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=41)
        loops = 300
        v_minmax = 5
        bins = 4
        error_tol = 0.016
        self.check_per_particle(
            N, kT, kT2, gamma2, loops, v_minmax, bins, error_tol)

    def setup_diff_mass_rinertia(self, p):
        if espressomd.has_features("MASS"):
            p.mass = 0.5
        if espressomd.has_features("ROTATION"):
            p.rotation = [1, 1, 1]
            # Make sure rinertia does not change diff coeff
            if espressomd.has_features("ROTATIONAL_INERTIA"):
                p.rinertia = [0.4, 0.4, 0.4]

    def test_06__diffusion(self):
        """This tests rotational and translational diffusion coeff via Green-Kubo"""
        system = self.system

        kT = 1.37
        dt = 0.1
        system.time_step = dt

        # Translational gamma. We cannot test per-component, if rotation is on,
        # because body and space frames become different.
        gamma = 3.1

        # Rotational gamma
        gamma_rot_i = 4.7
        gamma_rot_a = [4.2, 1, 1.2]

        # If we have langevin per particle:
        # per particle kT
        per_part_kT = 1.6
        # Translation
        per_part_gamma = 1.63
        # Rotational
        per_part_gamma_rot_i = 2.6
        per_part_gamma_rot_a = [2.4, 3.8, 1.1]

        # Particle with global thermostat params
        p_global = system.part.add(pos=(0, 0, 0))
        # Make sure, mass doesn't change diff coeff
        self.setup_diff_mass_rinertia(p_global)

        # particle specific gamma, kT, and both
        if espressomd.has_features("LANGEVIN_PER_PARTICLE"):
            p_gamma = system.part.add(pos=(0, 0, 0))
            self.setup_diff_mass_rinertia(p_gamma)
            if espressomd.has_features("PARTICLE_ANISOTROPY"):
                p_gamma.gamma = per_part_gamma, per_part_gamma, per_part_gamma
                if espressomd.has_features("ROTATION"):
                    p_gamma.gamma_rot = per_part_gamma_rot_a
            else:
                p_gamma.gamma = per_part_gamma
                if espressomd.has_features("ROTATION"):
                    p_gamma.gamma_rot = per_part_gamma_rot_i

            p_kT = system.part.add(pos=(0, 0, 0))
            self.setup_diff_mass_rinertia(p_kT)
            p_kT.temp = per_part_kT

            p_both = system.part.add(pos=(0, 0, 0))
            self.setup_diff_mass_rinertia(p_both)
            p_both.temp = per_part_kT
            if espressomd.has_features("PARTICLE_ANISOTROPY"):
                p_both.gamma = per_part_gamma, per_part_gamma, per_part_gamma
                if espressomd.has_features("ROTATION"):
                    p_both.gamma_rot = per_part_gamma_rot_a
            else:
                p_both.gamma = per_part_gamma
                if espressomd.has_features("ROTATION"):
                    p_both.gamma_rot = per_part_gamma_rot_i

        # Thermostat setup
        if espressomd.has_features("ROTATION"):
            if espressomd.has_features("PARTICLE_ANISOTROPY"):
                # particle anisotropy and rotation
                system.thermostat.set_langevin(
                    kT=kT, gamma=gamma, gamma_rotation=gamma_rot_a, seed=41)
            else:
                # Rotation without particle anisotropy
                system.thermostat.set_langevin(
                    kT=kT, gamma=gamma, gamma_rotation=gamma_rot_i, seed=41)
        else:
            # No rotation
            system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=41)

        system.cell_system.skin = 0.4
        system.integrator.run(100)

        # Correlators
        vel_obs = {}
        omega_obs = {}
        corr_vel = {}
        corr_omega = {}
        all_particles = [p_global]
        if espressomd.has_features("LANGEVIN_PER_PARTICLE"):
            all_particles.append(p_gamma)
            all_particles.append(p_kT)
            all_particles.append(p_both)

        # linear vel
        vel_obs = ParticleVelocities(ids=system.part[:].id)
        corr_vel = Correlator(
            obs1=vel_obs, tau_lin=10, tau_max=1.4, delta_N=2,
            corr_operation="componentwise_product", compress1="discard1")
        system.auto_update_accumulators.add(corr_vel)
        # angular vel
        if espressomd.has_features("ROTATION"):
            omega_obs = ParticleBodyAngularVelocities(ids=system.part[:].id)
            corr_omega = Correlator(
                obs1=omega_obs, tau_lin=10, tau_max=1.5, delta_N=2,
                corr_operation="componentwise_product", compress1="discard1")
            system.auto_update_accumulators.add(corr_omega)

        system.integrator.run(80000)

        system.auto_update_accumulators.remove(corr_vel)
        corr_vel.finalize()
        if espressomd.has_features("ROTATION"):
            system.auto_update_accumulators.remove(corr_omega)
            corr_omega.finalize()

        # Verify diffusion
        # Translation
        # Cast gammas to vector, to make checks independent of
        # PARTICLE_ANISOTROPY
        gamma = np.ones(3) * gamma
        per_part_gamma = np.ones(3) * per_part_gamma
        self.verify_diffusion(p_global, corr_vel, kT, gamma)
        if espressomd.has_features("LANGEVIN_PER_PARTICLE"):
            self.verify_diffusion(p_gamma, corr_vel, kT, per_part_gamma)
            self.verify_diffusion(p_kT, corr_vel, per_part_kT, gamma)
            self.verify_diffusion(
                p_both, corr_vel, per_part_kT, per_part_gamma)

        # Rotation
        if espressomd.has_features("ROTATION"):
            # Decide on effective gamma rotation, since for rotation it is
            # direction dependent
            eff_gamma_rot = None
            if espressomd.has_features("PARTICLE_ANISOTROPY"):
                eff_gamma_rot = gamma_rot_a
                eff_per_part_gamma_rot = per_part_gamma_rot_a
            else:
                eff_gamma_rot = gamma_rot_i * np.ones(3)
                eff_per_part_gamma_rot = per_part_gamma_rot_i * np.ones(3)

            self.verify_diffusion(p_global, corr_omega, kT, eff_gamma_rot)
            if espressomd.has_features("LANGEVIN_PER_PARTICLE"):
                self.verify_diffusion(
                    p_gamma, corr_omega, kT, eff_per_part_gamma_rot)
                self.verify_diffusion(
                    p_kT, corr_omega, per_part_kT, eff_gamma_rot)
                self.verify_diffusion(p_both, corr_omega,
                                      per_part_kT, eff_per_part_gamma_rot)

    def verify_diffusion(self, p, corr, kT, gamma):
        """Verify diffusion coeff.

           p: particle, corr: dict containing correlator with particle as key,
           kT=kT, gamma=gamma as 3 component vector.
        """
        c = corr
        # Integral of vacf via Green-Kubo D = int_0^infty <v(t_0)v(t_0+t)> dt
        # (or 1/3, since we work componentwise)
        acf = c.result()
        tau = c.lag_times()

        # Integrate with trapezoidal rule
        for i in range(3):
            I = np.trapz(acf[:, p.id, i], tau)
            ratio = I / (kT / gamma[i])
            self.assertAlmostEqual(ratio, 1., delta=0.07)

    @utx.skipIfMissingFeatures("VIRTUAL_SITES")
    def test_07__virtual(self):
        system = self.system
        system.time_step = 0.01

        virtual = system.part.add(pos=[0, 0, 0], virtual=True, v=[1, 0, 0])
        physical = system.part.add(pos=[0, 0, 0], virtual=False, v=[1, 0, 0])

        system.thermostat.set_langevin(
            kT=0, gamma=1, gamma_rotation=1., act_on_virtual=False, seed=41)

        system.integrator.run(0)

        np.testing.assert_almost_equal(np.copy(virtual.f), [0, 0, 0])
        np.testing.assert_almost_equal(np.copy(physical.f), [-1, 0, 0])

        system.thermostat.set_langevin(
            kT=0, gamma=1, gamma_rotation=1., act_on_virtual=True, seed=41)
        system.integrator.run(0)

        np.testing.assert_almost_equal(np.copy(virtual.f), [-1, 0, 0])
        np.testing.assert_almost_equal(np.copy(physical.f), [-1, 0, 0])

    def test_08__noise_correlation(self):
        """Checks that the Langevin noise is uncorrelated"""

        system = self.system
        system.time_step = 0.01
        system.cell_system.skin = 0.1
        kT = 3.2
        system.thermostat.set_langevin(kT=kT, gamma=5.1, seed=17)
        system.part.add(id=(1, 2), pos=np.zeros((2, 3)))
        steps = int(2E5)
        error_delta = 0.04
        self.check_noise_correlation(kT, steps, error_delta)


if __name__ == "__main__":
    ut.main()
