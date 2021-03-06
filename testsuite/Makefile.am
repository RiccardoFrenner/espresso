# Copyright (C) 2012,2013,2014 The ESPResSo project
# Copyright (C) 2007,2008,2009,2010,2011 Olaf Lenz, Axel Arnold
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
# alphabetically sorted list of test scripts
tests = \
	analysis.tcl \
	angle.tcl \
	bonded_coulomb.tcl \
	collision-detection-centers.tcl \
	collision-detection-poc.tcl \
	comforce.tcl \
	comfixed.tcl \
	command_syntax.tcl \
	constraints.tcl \
	constraints_reflecting.tcl \
	correlation.tcl \
	correlation_checkpoint.tcl \
	constraints_rhomboid.tcl \
	coulomb_cloud_wall.tcl \
	dh.tcl \
	dielectric.tcl \
	ek_eof_one_species_x.tcl \
	ek_eof_one_species_y.tcl \
	ek_eof_one_species_z.tcl \
	ek_eof_one_species_x_nonlinear.tcl \
	ek_eof_one_species_y_nonlinear.tcl \
	ek_eof_one_species_z_nonlinear.tcl \
	ek_electrostatics_coupling_x.tcl \
	ek_electrostatics_coupling_y.tcl \
	ek_electrostatics_coupling_z.tcl \
	ek_reaction.tcl \
	el2d.tcl \
	el2d_nonneutral.tcl \
	engine_langevin.tcl \
	engine_lb.tcl \
	engine_lbgpu.tcl \
	exclusions.tcl \
	external_potential.tcl \
	fene.tcl \
	gb.tcl \
	ghmc.tcl \
	harm.tcl \
	quartic.tcl \
	iccp3m.tcl \
	immersed_boundary.tcl \
	immersed_boundary_gpu.tcl \
	intpbc.tcl \
	intppbc.tcl \
	kinetic.tcl \
	langevin.tcl \
	layered.tcl \
	lb.tcl \
	lb_fluid_coupling.tcl \
	lb_fluid_coupling_gpu.tcl \
	lb_gpu.tcl \
	lb_planar.tcl \
	lb_planar_gpu.tcl \
	lb_planar_embedded_particles.tcl \
	lb_planar_embedded_particles_gpu.tcl \
	lb_stokes_sphere.tcl \
	lb_stokes_sphere_gpu.tcl \
	lees_edwards.tcl \
	lj.tcl \
	lj-cos.tcl \
	lj-generic.tcl \
	madelung.tcl \
	maggs.tcl \
	magnetic-field.tcl \
	mass.tcl \
	mass-and-rinertia.tcl \
	mdlc.tcl \
	minimize_energy.tcl \
	mmm1d.tcl \
	mmm1dgpu.tcl \
	ewaldgpu.tcl \
	npt.tcl \
	nsquare.tcl \
	nve_pe.tcl \
	object_in_fluid.tcl \
	object_in_fluid_gpu.tcl \
	observable.tcl \
	p3m.tcl \
	p3m_gpu.tcl \
	p3m_magnetostatics.tcl \
	p3m_magnetostatics2.tcl \
	p3m_simple_noncubic.tcl \
	pdb_parser.tcl \
	rotation.tcl \
	rotation_per_particle.tcl \
	sc_fluid_gpu.tcl \
	sd_ewald.tcl \
	sd_two_spheres.tcl \
	sd_thermalization.tcl \
	tabulated.tcl \
        tunable_slip.tcl \
        uwerr.tcl \
	virtual-sites.tcl 
# please keep the alphabetic ordering of the above list!

deactivated_tests =

# add data files for the tests here
EXTRA_DIST = $(tests) $(deactivated_tests) \
	tests_common.tcl \
	analysis_system.data.00.gz analysis_system.data.01.gz \
	analysis_system.data.02.gz analysis_system.data.03.gz \
	analysis_system.data.04.gz analysis_system.data.05.gz \
	analysis_system.data.06.gz analysis_system.data.07.gz \
	analysis_system.data.08.gz analysis_system.data.09.gz \
	analysis_system.data.10.gz \
	angle_cosine.data angle_cossquare.data angle_harmonic.data \
	comforce_system.data comfixed_system.data \
	constraints_system.data \
	coulomb_cloud_wall_system.data \
        p3m_magnetostatics.data \
        p3m_magnetostatics2_system.data p3m_magnetostatics2_expected.data \
	p3m_system.data p3m_system_gpu.data \
	el2d_system.data el2d_system_die.data \
	mdlc_system.data mdlc_expected_energy.data mdlc_expected_force_torque.data \
	mmm1d_system.data dh_system.data \
	lj_system.data lj-cos_system.data \
	gb_system.data uwerr.data \
	tabulated_system.data lj1.tab lj2.tab lj3.tab \
	fene_system.data.gz harm_system.data.gz \
	harmonic_potential.dat \
	intpbc_system.data.gz intppbc.tcl intppbc_system.data.gz \
	npt_lj_system.data \
	thermostat.data thermostat_rot.data \
	mass_system.data \
	lb_system.data \
	object_in_fluid_system-final.data \
	object_in_fluid_system-init.data \
        object_in_fluid_system-nodes.data \
	object_in_fluid_system-triangles.data \
	pdb_parser.itp.dat \
	pdb_parser.pdb.dat \
	pe_micelle.tcl pe_micelle.data \
	maggs_correct_rdf.data \
	ek_reaction_density.vtk \
	engine_lb.vtk \
	engine_lbgpu_2pt.vtk \
	engine_lbgpu_3pt.vtk

# run the testsuite
check-local: runtest.sh
	if test -n "$(processors)"; then \
	  @builddir@/runtest.sh -p "${processors}" $(tests); \
	else \
	  @builddir@/runtest.sh $(tests); \
	fi

CLEANFILES = runtest.log

DISTCLEANFILES = \
	fftw3_1d_wisdom_*.file \
	dfftw3_1d_wisdom_*.file \
	lattice_*.dat \
	correlation_checkpoint_c?.bin \
	correlation_checkpoint_c?.txt \
	sd_thermalization.log
