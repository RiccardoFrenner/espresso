/*
 * Copyright (C) 2020 The ESPResSo project
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef LB_WALBERLA_H
#define LB_WALBERLA_H

/**
 * @file
 * @ref walberla::LBWalberlaImpl implements the interface of the LB
 * waLBerla bridge. It is a templated class that is specialized by lattice
 * models created by lbmpy (see <tt>maintainer/walberla_kernels</tt>).
 */

#include "blockforest/Initialization.h"
#include "blockforest/StructuredBlockForest.h"
#include "blockforest/communication/UniformBufferedScheme.h"
#include "boundary/BoundaryHandling.h"
#include "core/SharedFunctor.h"
#include "field/GhostLayerField.h"
#include "field/adaptors/GhostLayerFieldAdaptor.h"
#include "field/vtk/FlagFieldCellFilter.h"
#include "field/vtk/VTKWriter.h"
#include "lbm/lattice_model/CollisionModel.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/vtk/all.h"
#include "timeloop/SweepTimeloop.h"

#include "pe/basic.h"
#include "pe/rigidbody/BodyStorage.h"
#include "pe/statistics/BodyStatistics.h"
#include "pe/utility/DestroyBody.h"

// TODO: Remove unused headers
#include "pe_coupling/mapping/all.h"
#include "pe_coupling/momentum_exchange_method/all.h"
#include "pe_coupling/utility/all.h"

#include "core/mpi/Environment.h"
#include "core/mpi/MPIManager.h"
#include "core/mpi/MPITextFile.h"
#include "core/mpi/Reduce.h"

#include "domain_decomposition/SharedSweep.h"

#include "field/AddToStorage.h"
#include "field/FlagField.h"
#include "field/adaptors/AdaptorCreators.h"
#include "field/communication/PackInfo.h"
#include "lbm/boundary/NoSlip.h"
#include "lbm/boundary/UBB.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/Adaptors.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/PdfField.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/sweeps/CellwiseSweep.h"

#include "stencil/D3Q27.h"

#include "timeloop/SweepTimeloop.h"

#include "LBWalberlaBase.hpp"
#include "PE_Parameters.hpp"
#include "ResetForce.hpp"
#include "walberla_utils.hpp"

#include <cassert>
#include <utility>
#include <utils/Vector.hpp>
#include <utils/interpolation/bspline_3d.hpp>
#include <utils/math/make_lin_space.hpp>

#include <boost/optional.hpp>
#include <boost/tuple/tuple.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace walberla {

// Flags marking fluid and boundaries
const FlagUID Fluid_flag("fluid");
const FlagUID UBB_flag("velocity bounce back");
const FlagUID MO_BB_Flag("moving obstacle BB");
const FlagUID FormerMO_Flag("former moving obstacle");

/** Class that runs and controls the LB on WaLBerla
 */
template <typename LatticeModel> class LBWalberlaImpl : public LBWalberlaBase {

protected:
  // Type definitions
  using VectorField = GhostLayerField<real_t, 3>;
  using FlagField = walberla::FlagField<walberla::uint8_t>;
  using PdfField = lbm::PdfField<LatticeModel>;
  using BodyField = GhostLayerField<pe::BodyID, 1>;
  static uint_t const FieldGhostLayers = 1; // Same value as used in BodyField
  /** Velocity boundary conditions */
  using UBB = lbm::UBB<LatticeModel, uint8_t, true, true>;
  typedef pe_coupling::SimpleBB<LatticeModel, FlagField> MO_BB;

  /** Boundary handling */
  using Boundaries =
      BoundaryHandling<FlagField, typename LatticeModel::Stencil, UBB, MO_BB>;

  // this uses the ´EquilibriumAndNonEquilibriumReconstructor´
  // all options:
  // ´EquilibriumAndNonEquilibriumReconstructor´,
  // ´EquilibriumReconstructor´ and
  // ´ExtrapolationReconstructor´.
  typedef pe_coupling::EquilibriumAndNonEquilibriumReconstructor<
      LatticeModel, Boundaries,
      pe_coupling::SphereNormalExtrapolationDirectionFinder>
      Reconstructor;

  // Adaptors
  using VelocityAdaptor = typename lbm::Adaptor<LatticeModel>::VelocityVector;

  // All currently used physics engine (pe) bodies
  using BodyTypeTuple = std::tuple<pe::Sphere, pe::Plane>;

  /** VTK writers that are executed automatically */
  std::map<std::string, std::pair<std::shared_ptr<vtk::VTKOutput>, bool>>
      m_vtk_auto;
  /** VTK writers that are executed manually */
  std::map<std::string, std::shared_ptr<vtk::VTKOutput>> m_vtk_manual;

  // Member variables
  Utils::Vector3i m_grid_dimensions;
  int m_n_ghost_layers;

  // Block data access handles
  BlockDataID m_pdf_field_id;
  BlockDataID m_flag_field_id;

  BlockDataID m_last_applied_force_field_id;
  BlockDataID m_force_to_be_applied_id;

  BlockDataID m_velocity_adaptor_id;

  BlockDataID m_boundary_handling_id;

  // Stores a pointer (pe::BodyID) inside each flagged moving boundary cell to
  // the containing particle
  BlockDataID m_body_field_id;

  using Communicator = blockforest::communication::UniformBufferedScheme<
      typename stencil::D3Q27>;
  std::shared_ptr<Communicator> m_communication;

  /** Block forest */
  std::shared_ptr<blockforest::StructuredBlockForest> m_blocks;

  std::shared_ptr<timeloop::SweepTimeloop> m_time_loop;

  // MPI
  std::shared_ptr<mpi::Environment> m_env;

  // Lattice model
  std::shared_ptr<LatticeModel> m_lattice_model;

  // ResetForce sweep + external force handling
  std::shared_ptr<ResetForce<PdfField, VectorField>> m_reset_force;

  // PE private members
  real_t m_pe_sync_overlap;
  PE_Parameters m_pe_parameters;

  // Global particle storage (for very large particles), stored on all processes
  std::shared_ptr<pe::BodyStorage> m_global_body_storage;

  // Storage handle for pe particles
  BlockDataID m_body_storage_id;

  // coarse and fine collision detection
  BlockDataID m_ccd_id;
  BlockDataID m_fcd_id;

  // pe time integrator
  std::shared_ptr<pe::cr::DEM> m_cr;

  // storage for force/torque to average over two timesteps
  shared_ptr<pe_coupling::BodiesForceTorqueContainer>
      m_bodies_force_torque_container_1;
  shared_ptr<pe_coupling::BodiesForceTorqueContainer>
      m_bodies_force_torque_container_2;
  shared_ptr<pe_coupling::ForceTorqueOnBodiesScaler> m_force_scaler;
  std::function<void(void)> m_store_force_torque_in_cont_1;
  std::function<void(void)> m_set_force_torque_on_bodies_from_cont_2;
  std::function<void(void)> m_set_force_scaling_factor_to_half;

  // calculates particle information on call
  // std::shared_ptr<pe::BodyStatistics> m_bodyStats;

  // view on particles to access them via their uid
  std::map<std::uint64_t, pe::BodyID> m_pe_particles;

  // force/torque is reset after each time step, so we need to store them here
  std::map<std::uint64_t, Vector3<real_t>> m_particle_forces;
  std::map<std::uint64_t, Vector3<real_t>> m_particle_torques;
  std::function<void(void)> m_save_force_torque;

  bool m_is_pe_initialized;
  std::function<void(void)> m_pe_sync_call;
  shared_ptr<pe_coupling::SphereNormalExtrapolationDirectionFinder>
      m_extrapolation_finder;
  shared_ptr<Reconstructor> m_reconstructor;

  size_t stencil_size() const override {
    return static_cast<size_t>(LatticeModel::Stencil::Size);
  }

  // Boundary handling
  class LBBoundaryHandling {
  public:
    LBBoundaryHandling(BlockDataID const &flag_field_id,
                       BlockDataID const &pdf_field_id,
                       BlockDataID const &body_field_id)
        : m_flag_field_id(flag_field_id), m_pdf_field_id(pdf_field_id),
          m_body_field_id(body_field_id) {}

    Boundaries *operator()(IBlock *const block,
                           StructuredBlockStorage const *const storage) const {
      WALBERLA_ASSERT_NOT_NULLPTR(block);
      WALBERLA_ASSERT_NOT_NULLPTR(storage);

      auto *flagField = block->template getData<FlagField>(m_flag_field_id);
      PdfField *pdfField = block->template getData<PdfField>(m_pdf_field_id);
      auto *bodyField = block->template getData<BodyField>(m_body_field_id);

      auto const fluid = flagField->flagExists(Fluid_flag)
                             ? flagField->getFlag(Fluid_flag)
                             : flagField->registerFlag(Fluid_flag);

      auto *handling = new Boundaries(
          "moving obstacle boundary handling", flagField, fluid,
          UBB("velocity bounce back", UBB_flag, pdfField, nullptr),
          MO_BB("MO", MO_BB_Flag, pdfField, flagField, bodyField, fluid,
                *storage, *block));

      handling->fillWithDomain(FieldGhostLayers);

      return handling;
    }

  private:
    BlockDataID const m_flag_field_id;
    BlockDataID const m_pdf_field_id;
    BlockDataID const m_body_field_id;

  }; // class LBBoundaryHandling

private:
  bool using_moving_obstacles() { return m_pe_parameters.use_moving_obstacles; }

  bool is_pe_initialized() { return m_is_pe_initialized; }

  void init_particle_force_averaging() {
    m_bodies_force_torque_container_1 =
        make_shared<pe_coupling::BodiesForceTorqueContainer>(m_blocks,
                                                             m_body_storage_id);
    m_store_force_torque_in_cont_1 =
        std::bind(&pe_coupling::BodiesForceTorqueContainer::store,
                  m_bodies_force_torque_container_1);
    m_bodies_force_torque_container_2 =
        make_shared<pe_coupling::BodiesForceTorqueContainer>(m_blocks,
                                                             m_body_storage_id);
    m_set_force_torque_on_bodies_from_cont_2 =
        std::bind(&pe_coupling::BodiesForceTorqueContainer::setOnBodies,
                  m_bodies_force_torque_container_2);
    m_force_scaler = make_shared<pe_coupling::ForceTorqueOnBodiesScaler>(
        m_blocks, m_body_storage_id, real_t(1));
    m_set_force_scaling_factor_to_half =
        std::bind(&pe_coupling::ForceTorqueOnBodiesScaler::resetScalingFactor,
                  m_force_scaler, real_t(0.5));
  }

  void init_pe_fields() {
    // add body field
    m_body_field_id = field::addToStorage<BodyField>(m_blocks, "body field",
                                                     nullptr, field::zyxf);
  }

  void init_body_synchronization() {
    if (!m_pe_parameters.sync_shadow_owners) {
      m_pe_sync_call = std::bind(
          pe::syncNextNeighbors<BodyTypeTuple>,
          std::ref(m_blocks->getBlockForest()), m_body_storage_id,
          static_cast<WcTimingTree *>(nullptr), m_pe_sync_overlap, false);
    } else {
      m_pe_sync_call = std::bind(
          pe::syncShadowOwners<BodyTypeTuple>,
          std::ref(m_blocks->getBlockForest()), m_body_storage_id,
          static_cast<WcTimingTree *>(nullptr), m_pe_sync_overlap, false);
    }
  }

  void init_boundary_handling() {
    assert(is_pe_initialized());

    // add boundary handling
    m_boundary_handling_id = m_blocks->addStructuredBlockData<Boundaries>(
        LBBoundaryHandling(m_flag_field_id, m_pdf_field_id, m_body_field_id),
        "MO boundary handling");

    // method for restoring PDFs in cells previously occupied by pe bodies
    m_extrapolation_finder =
        make_shared<pe_coupling::SphereNormalExtrapolationDirectionFinder>(
            m_blocks, m_body_field_id);
    m_reconstructor =
        make_shared<Reconstructor>(m_blocks, m_boundary_handling_id,
                                   m_body_field_id, *m_extrapolation_finder);
  }

  void init_time_loop(uint_t const timesteps) {
    // The following briefly describes the steps of the timeloop.
    // Expressions in brackets only appear when using moving obstacles.
    // - sweep: Reset force fields
    // - sweep: MO Boundary Handling
    // - sweep: LB stream & collide
    // - (sweep: Body Mapping)
    // - (sweep: PDF Restore)
    // - (afterFunc: Force storing)
    // - (afterFunc: Force setting)
    // - (afterFunc: Force averaging)
    // - (afterFunc: Force scaling adjustment)
    // - (afterFunc: Swap FT container)
    // - (afterFunc: Constant global forces)
    // - (afterFunc: pe Time Step)
    // - afterFunc: communication
    // Communication needs to happen at the end, because espresso relies on the
    // values beeing up to date also on the ghost layers.

    // create the timeloop
    m_time_loop =
        make_shared<SweepTimeloop>(m_blocks->getBlockStorage(), timesteps);

    m_time_loop->add() << timeloop::Sweep(makeSharedSweep(m_reset_force),
                                          "Reset force fields");

    // add boundary handling sweep
    m_time_loop->add() << timeloop::Sweep(
        Boundaries::getBlockSweep(m_boundary_handling_id),
        "MO Boundary Handling");

    m_time_loop->add() << timeloop::Sweep(
        typename LatticeModel::Sweep(m_pdf_field_id), "LB stream & collide");

    if (using_moving_obstacles()) {
      // Averaging the force/torque over two time steps is said to damp
      // oscillations of the interaction force/torque. See Ladd - " Numerical
      // simulations of particulate suspensions via a discretized Boltzmann
      // equation. Part 1. Theoretical foundation", 1994, p. 302
      if (m_pe_parameters.average_force_torque_over_two_timesteps) {
        // store force/torque from hydrodynamic interactions in container1
        m_time_loop->addFuncAfterTimeStep(m_store_force_torque_in_cont_1,
                                          "Force storing");

        // set force/torque from previous time step (in container2)
        m_time_loop->addFuncAfterTimeStep(
            m_set_force_torque_on_bodies_from_cont_2, "Force setting");

        // average the force/torque by scaling it with factor 1/2 (except in
        // first timestep, there it is 1, which it is initially)
        m_time_loop->addFuncAfterTimeStep(
            SharedFunctor<pe_coupling::ForceTorqueOnBodiesScaler>(
                m_force_scaler),
            "Force averaging");
        m_time_loop->addFuncAfterTimeStep(m_set_force_scaling_factor_to_half,
                                          "Force scaling adjustment");

        // swap containers
        m_time_loop->addFuncAfterTimeStep(
            pe_coupling::BodyContainerSwapper(
                m_bodies_force_torque_container_1,
                m_bodies_force_torque_container_2),
            "Swap FT container");
      }

      // save force/torque before overriden by global force or reset by pe step
      m_time_loop->addFuncAfterTimeStep(m_save_force_torque,
                                        "Save force/torque");

      // add constant global forces
      for (auto &&t : m_pe_parameters.constant_global_forces) {
        m_time_loop->addFuncAfterTimeStep(
            pe_coupling::ForceOnBodiesAdder(m_blocks, m_body_storage_id,
                                            to_vector3(t.first)),
            t.second);
      }
      // integrates particles and resets forces and torques afterwards
      m_time_loop->addFuncAfterTimeStep(
          pe_coupling::TimeStep(m_blocks, m_body_storage_id, *m_cr,
                                m_pe_sync_call, real_t(1),
                                m_pe_parameters.num_pe_sub_cycles),
          "pe Time Step");

      // sweep for updating the pe body mapping into the LBM simulation
      m_time_loop->add() << timeloop::Sweep(
          pe_coupling::BodyMapping<LatticeModel, Boundaries>(
              m_blocks, m_pdf_field_id, m_boundary_handling_id,
              m_body_storage_id, m_global_body_storage, m_body_field_id,
              MO_BB_Flag, FormerMO_Flag, pe_coupling::selectRegularBodies),
          "Body Mapping");

      // sweep for restoring PDFs in cells previously occupied by pe bodies
      m_time_loop->add() << timeloop::Sweep(
          pe_coupling::PDFReconstruction<LatticeModel, Boundaries,
                                         Reconstructor>(
              m_blocks, m_pdf_field_id, m_boundary_handling_id,
              m_body_storage_id, m_global_body_storage, m_body_field_id,
              *m_reconstructor, FormerMO_Flag, Fluid_flag),
          "PDF Restore");
    }

    m_time_loop->addFuncAfterTimeStep(*m_communication, "communication");
  }

  void init_pe() {
    init_pe_fields();

    m_global_body_storage = make_shared<pe::BodyStorage>();

    // Init pe body type ids
    pe::SetBodyTypeIDs<BodyTypeTuple>::execute();

    m_body_storage_id = m_blocks->addBlockData(
        pe::createStorageDataHandling<BodyTypeTuple>(), "pe Body Storage");

    // coarse and fine collision detection
    m_ccd_id =
        m_blocks->addBlockData(pe::ccd::createHashGridsDataHandling(
                                   m_global_body_storage, m_body_storage_id),
                               "CCD");
    m_fcd_id = m_blocks->addBlockData(
        pe::fcd::createGenericFCDDataHandling<
            BodyTypeTuple, pe::fcd::AnalyticCollideFunctor>(),
        "FCD");

    // set up collision response, here DEM solver
    // cr::HCSITS is for hard contacts, cr::DEM for soft contacts
    // TODO: Has many parameters which could be changed
    m_cr = make_shared<pe::cr::DEM>(
        m_global_body_storage, m_blocks->getBlockStoragePointer(),
        m_body_storage_id, m_ccd_id, m_fcd_id, nullptr);

    // set up body synchronization procedure
    init_body_synchronization();

    if (m_pe_parameters.average_force_torque_over_two_timesteps) {
      init_particle_force_averaging();
    }

    // define function for saving force/torque before reset through pe time step
    m_save_force_torque = [this]() {
      for (auto blockIt = m_blocks->begin(); blockIt != m_blocks->end();
           ++blockIt) {

        for (auto bodyIt = pe::BodyIterator::begin(*blockIt, m_body_storage_id);
             bodyIt != pe::BodyIterator::end(); ++bodyIt) {

          auto const &force = bodyIt->getForce();
          auto const &torque = bodyIt->getTorque();
          m_particle_forces[bodyIt->getID()] = force;
          m_particle_torques[bodyIt->getID()] = torque;
        }
      }
    };

    m_is_pe_initialized = true;
  }

  void init_lb_fields() {
    // Init and register force fields
    m_last_applied_force_field_id = field::addToStorage<VectorField>(
        m_blocks, "force field", real_t{0}, field::fzyx, m_n_ghost_layers);
    m_force_to_be_applied_id = field::addToStorage<VectorField>(
        m_blocks, "force field", real_t{0}, field::fzyx, m_n_ghost_layers);

    // Init and register flag field (fluid/boundary)
    m_flag_field_id = field::addFlagFieldToStorage<FlagField>(
        m_blocks, "flag field", m_n_ghost_layers);
  }

  void init_blockforest(Utils::Vector3i const &n_blocks,
                        Utils::Vector3i const &n_cells_per_block,
                        double const lb_cell_size,
                        Utils::Vector3i const &n_processes) {

    // Running a simulation with periodic boundaries needs at least three
    // blocks in each direction of periodicity
    Vector3<bool> periodicity(true, true, true);
    for (int i = 0; i < 3; i++) {
      if (periodicity[i] && n_blocks[i] < 3) {
        // TODO: Produces a lot of errors in LBWalberla_test
        // throw std::runtime_error("Direction " + std::to_string(i) +
        //                          " needs at least three blocks but only " +
        //                          std::to_string(n_blocks[i]) + " where
        //                          given");
      }
    }

    m_blocks = blockforest::createUniformBlockGrid(
        uint_c(n_blocks[0]),          // blocks in x direction
        uint_c(n_blocks[1]),          // blocks in y direction
        uint_c(n_blocks[2]),          // blocks in z direction
        uint_c(n_cells_per_block[0]), // cells per block in x direction
        uint_c(n_cells_per_block[1]), // cells per block in y direction
        uint_c(n_cells_per_block[2]), // cells per block in z direction
        1,                            // Lattice constant
        uint_c(n_processes[0]),       // cpus in x direction
        uint_c(n_processes[1]),       // cpus in y direction
        uint_c(n_processes[2]),       // cpus in z direction
        true, true, true);            // periodicity
  }

public:
  LBWalberlaImpl(Utils::Vector3i const &n_blocks,
                 Utils::Vector3i const &n_cells_per_block,
                 double const lb_cell_size, Utils::Vector3i const &n_processes,
                 int n_ghost_layers, PE_Parameters pe_params = PE_Parameters())
      : m_n_ghost_layers(n_ghost_layers),
        m_pe_parameters(std::move(pe_params)) {

    if (m_n_ghost_layers <= 0)
      throw std::runtime_error("At least one ghost layer must be used");

    init_blockforest(n_blocks, n_cells_per_block, lb_cell_size, n_processes);

    init_lb_fields();

    // Set overlap for pe particle syncronization routine
    // overlap of 1.5 * lattice grid spacing is needed for correct mapping
    // see
    // https://www.walberla.net/doxygen/group__pe__coupling.html#reconstruction
    // for more information
    double const lattice_grid_spacing = 1.0; // aka. `Lattice constant` above
    m_pe_sync_overlap =
        m_pe_parameters.syncronization_overlap_factor * lattice_grid_spacing;

    // Init body statistics
    // m_bodyStats =
    //     std::make_shared<pe::BodyStatistics>(m_blocks, m_body_storage_id);
    // (*m_bodyStats)();
  }

  LBWalberlaImpl(double viscosity, double agrid, double tau,
                 const Utils::Vector3d &box_dimensions,
                 const Utils::Vector3i &node_grid, int n_ghost_layers,
                 PE_Parameters pe_params = PE_Parameters())
      : m_n_ghost_layers(n_ghost_layers),
        m_pe_parameters(std::move(pe_params)) {

    if (m_n_ghost_layers <= 0)
      throw std::runtime_error("At least one ghost layer must be used");

    for (int i = 0; i < 3; i++) {
      if (fabs(round(box_dimensions[i] / agrid) * agrid - box_dimensions[i]) /
              box_dimensions[i] >
          std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error(
            "Box length not commensurate with agrid in direction " +
            std::to_string(i));
      }
      m_grid_dimensions[i] = int(std::round(box_dimensions[i] / agrid));
      if (m_grid_dimensions[i] % node_grid[i] != 0) {
        printf("Grid dimension: %d, node grid %d\n", m_grid_dimensions[i],
               node_grid[i]);
        throw std::runtime_error(
            "LB grid dimensions and mpi node grid are not compatible.");
      }
    }

    Utils::Vector3i n_cells_per_block{int(m_grid_dimensions[0] / node_grid[0]),
                                      int(m_grid_dimensions[1] / node_grid[1]),
                                      int(m_grid_dimensions[2] / node_grid[2])};

    init_blockforest(node_grid, n_cells_per_block, agrid, node_grid);

    init_lb_fields();

    // Set overlap for pe particle syncronization routine
    // overlap of 1.5 * lattice grid spacing is needed for correct mapping
    // see
    // https://www.walberla.net/doxygen/group__pe__coupling.html#reconstruction
    // for more information
    double const lattice_grid_spacing = 1.0; // aka. `Lattice constant` above
    m_pe_sync_overlap =
        m_pe_parameters.syncronization_overlap_factor * lattice_grid_spacing;
  };

  void setup_with_valid_lattice_model(double density) {
    // Init and register pdf field
    m_pdf_field_id = lbm::addPdfFieldToStorage(
        m_blocks, "pdf field", *(m_lattice_model.get()),
        to_vector3(Utils::Vector3d{}), real_t(density), m_n_ghost_layers);

    // pe needs to be initialized even if it is not used.
    // The boundary handling would otherwise not work since it includes field
    // used for MO.
    init_pe();

    // Register boundary handling
    init_boundary_handling();

    // sets up the communication and registers pdf field and force field to it
    m_communication = std::make_shared<Communicator>(m_blocks);
    m_communication->addPackInfo(
        std::make_shared<field::communication::PackInfo<PdfField>>(
            m_pdf_field_id));
    m_communication->addPackInfo(
        std::make_shared<field::communication::PackInfo<VectorField>>(
            m_last_applied_force_field_id));

    // Instance the sweep responsible for force double buffering and
    // external forces
    m_reset_force = std::make_shared<ResetForce<PdfField, VectorField>>(
        m_pdf_field_id, m_last_applied_force_field_id,
        m_force_to_be_applied_id);

    init_time_loop(1);

    // Register velocity access adapter (proxy)
    m_velocity_adaptor_id = field::addFieldAdaptor<VelocityAdaptor>(
        m_blocks, m_pdf_field_id, "velocity adaptor");

    // Synchronize ghost layers
    (*m_communication)();
  };

  std::shared_ptr<LatticeModel> get_lattice_model() { return m_lattice_model; };

  void integrate() override {
    m_time_loop->singleStep();

    // Handle VTK writers
    for (auto it = m_vtk_auto.begin(); it != m_vtk_auto.end(); ++it) {
      if (it->second.second)
        vtk::writeFiles(it->second.first)();
    }
  };

  void ghost_communication() override { (*m_communication)(); }

  template <typename Function>
  void interpolate_bspline_at_pos(Utils::Vector3d pos, Function f) const {
    Utils::Interpolation::bspline_3d<2>(
        pos, f, Utils::Vector3d{1.0, 1.0, 1.0}, // grid spacing
        Utils::Vector3d::broadcast(.5));        // offset (cell center)
  }

  // Velocity
  boost::optional<Utils::Vector3d>
  get_node_velocity(const Utils::Vector3i &node,
                    bool consider_ghosts = false) const override {
    boost::optional<bool> is_boundary =
        get_node_is_boundary(node, consider_ghosts);
    if (is_boundary)    // is info available locally
      if (*is_boundary) // is the node a boundary
        return get_node_velocity_at_boundary(node);
    auto const bc =
        get_block_and_cell(node, consider_ghosts, m_blocks, n_ghost_layers());
    if (!bc)
      return {};
    auto const &vel_adaptor =
        (*bc).block->template getData<VelocityAdaptor>(m_velocity_adaptor_id);
    return {to_vector3d(vel_adaptor->get((*bc).cell))};
  };
  bool set_node_velocity(const Utils::Vector3i &node,
                         const Utils::Vector3d &v) override {
    auto bc = get_block_and_cell(node, false, m_blocks, n_ghost_layers());
    if (!bc)
      return false;
    auto pdf_field = (*bc).block->template getData<PdfField>(m_pdf_field_id);
    const real_t density = pdf_field->getDensity((*bc).cell);
    pdf_field->setDensityAndVelocity(
        (*bc).cell, Vector3<real_t>{real_c(v[0]), real_c(v[1]), real_c(v[2])},
        density);
    return true;
  };

  boost::optional<Utils::Vector3d>
  get_velocity_at_pos(const Utils::Vector3d &pos,
                      bool consider_points_in_halo = false) const override {
    if (!consider_points_in_halo and !pos_in_local_domain(pos))
      return {};
    if (consider_points_in_halo and !pos_in_local_halo(pos))
      return {};
    Utils::Vector3d v{0.0, 0.0, 0.0};
    interpolate_bspline_at_pos(
        pos, [this, &v, pos](const std::array<int, 3> node, double weight) {
          // Nodes with zero weight might not be accessible, because they can be
          // outside ghost layers
          if (weight != 0) {
            auto res = get_node_velocity(
                Utils::Vector3i{{node[0], node[1], node[2]}}, true);
            if (!res) {
              printf("Pos: %g %g %g, Node %d %d %d, weight %g\n", pos[0],
                     pos[1], pos[2], node[0], node[1], node[2], weight);
              throw std::runtime_error("Access to LB velocity field failed.");
            }
            v += *res * weight;
          }
        });
    return {v};
  };

  // Local force
  bool add_force_at_pos(const Utils::Vector3d &pos,
                        const Utils::Vector3d &force) override {
    if (!pos_in_local_halo(pos))
      return false;
    auto force_at_node = [this, force](const std::array<int, 3> node,
                                       double weight) {
      auto const bc = get_block_and_cell(to_vector3i(node), true, m_blocks,
                                         n_ghost_layers());
      if (bc) {
        auto force_field = (*bc).block->template getData<VectorField>(
            m_force_to_be_applied_id);
        for (int i : {0, 1, 2})
          force_field->get((*bc).cell, i) += real_c(force[i] * weight);
      }
    };
    interpolate_bspline_at_pos(pos, force_at_node);
    return true;
  };

  boost::optional<Utils::Vector3d>
  get_node_force_to_be_applied(const Utils::Vector3i &node) const override {
    auto const bc = get_block_and_cell(node, true, m_blocks, n_ghost_layers());
    if (!bc)
      return {};

    auto const &force_field =
        (*bc).block->template getData<VectorField>(m_force_to_be_applied_id);
    return Utils::Vector3d{double_c(force_field->get((*bc).cell, uint_t(0u))),
                           double_c(force_field->get((*bc).cell, uint_t(1u))),
                           double_c(force_field->get((*bc).cell, uint_t(2u)))};
  };

  bool set_node_last_applied_force(Utils::Vector3i const &node,
                                   Utils::Vector3d const &force) override {
    auto bc = get_block_and_cell(node, false, m_blocks, n_ghost_layers());
    if (!bc)
      return false;

    auto force_field = (*bc).block->template getData<VectorField>(
        m_last_applied_force_field_id);
    for (uint_t f = 0u; f < 3u; ++f) {
      force_field->get((*bc).cell, f) = real_c(force[f]);
    }

    return true;
  };

  boost::optional<Utils::Vector3d>
  get_node_last_applied_force(const Utils::Vector3i &node,
                              bool consider_ghosts = false) const override {
    auto const bc =
        get_block_and_cell(node, consider_ghosts, m_blocks, n_ghost_layers());
    if (!bc)
      return {};

    auto const force_field = (*bc).block->template getData<VectorField>(
        m_last_applied_force_field_id);
    return Utils::Vector3d{double_c(force_field->get((*bc).cell, uint_t(0u))),
                           double_c(force_field->get((*bc).cell, uint_t(1u))),
                           double_c(force_field->get((*bc).cell, uint_t(2u)))};
    ;
  };

  // Population
  bool set_node_pop(const Utils::Vector3i &node,
                    std::vector<double> const &population) override {
    auto bc = get_block_and_cell(node, false, m_blocks, n_ghost_layers());
    if (!bc)
      return false;

    auto pdf_field = (*bc).block->template getData<PdfField>(m_pdf_field_id);
    constexpr auto FSize = LatticeModel::Stencil::Size;
    assert(population.size() == FSize);
    for (uint_t f = 0u; f < FSize; ++f) {
      pdf_field->get((*bc).cell, f) = real_c(population[f]);
    }

    return true;
  }

  boost::optional<std::vector<double>>
  get_node_pop(const Utils::Vector3i &node) const override {
    auto bc = get_block_and_cell(node, false, m_blocks, n_ghost_layers());
    if (!bc)
      return {boost::none};

    auto pdf_field = bc->block->template getData<PdfField>(m_pdf_field_id);
    constexpr auto FSize = LatticeModel::Stencil::Size;
    std::vector<double> population(FSize);
    for (uint_t f = 0u; f < FSize; ++f) {
      population[f] = double_c(pdf_field->get((*bc).cell, f));
    }

    return {population};
  }

  // Density
  bool set_node_density(const Utils::Vector3i &node, double density) override {
    auto bc = get_block_and_cell(node, false, m_blocks, n_ghost_layers());
    if (!bc)
      return false;

    auto pdf_field = (*bc).block->template getData<PdfField>(m_pdf_field_id);
    auto const &vel_adaptor =
        (*bc).block->template getData<VelocityAdaptor>(m_velocity_adaptor_id);
    Vector3<real_t> v = vel_adaptor->get((*bc).cell);

    pdf_field->setDensityAndVelocity((*bc).cell, v, real_c(density));

    return true;
  };

  boost::optional<double>
  get_node_density(const Utils::Vector3i &node) const override {
    auto bc = get_block_and_cell(node, false, m_blocks, n_ghost_layers());
    if (!bc)
      return {boost::none};

    auto pdf_field = (*bc).block->template getData<PdfField>(m_pdf_field_id);

    return {double_c(pdf_field->getDensity((*bc).cell))};
  };

  // Boundary related
  boost::optional<Utils::Vector3d>
  get_node_velocity_at_boundary(const Utils::Vector3i &node) const override {
    auto bc = get_block_and_cell(node, true, m_blocks, n_ghost_layers());
    if (!bc)
      return {boost::none};
    const Boundaries *boundary_handling =
        (*bc).block->template getData<Boundaries>(m_boundary_handling_id);
    boundary::BoundaryUID uid = boundary_handling->getBoundaryUID(UBB_flag);

    if (!boundary_handling->isBoundary((*bc).cell))
      return {boost::none};

    return {to_vector3d(
        boundary_handling->template getBoundaryCondition<UBB>(uid).getValue(
            (*bc).cell[0], (*bc).cell[1], (*bc).cell[2]))};
  };
  bool set_node_velocity_at_boundary(const Utils::Vector3i &node,
                                     const Utils::Vector3d &v) override {
    auto bc = get_block_and_cell(node, true, m_blocks, n_ghost_layers());
    if (!bc)
      return false;

    const typename UBB::Velocity velocity(real_c(v[0]), real_c(v[1]),
                                          real_c(v[2]));

    Boundaries *boundary_handling =
        (*bc).block->template getData<Boundaries>(m_boundary_handling_id);
    boundary_handling->forceBoundary(UBB_flag, bc->cell[0], bc->cell[1],
                                     bc->cell[2], velocity);
    return true;
  };
  boost::optional<Utils::Vector3d>
  get_node_boundary_force(const Utils::Vector3i &node) const override {
    auto bc = get_block_and_cell(node, true, m_blocks,
                                 n_ghost_layers()); // including ghosts
    if (!bc)
      return {boost::none};
    // Get boundary handling
    auto const &bh =
        (*bc).block->template getData<Boundaries>(m_boundary_handling_id);
    auto const &ff = (*bc).block->template getData<FlagField>(m_flag_field_id);
    try {
      if (!ff->isFlagSet((*bc).cell, ff->getFlag(UBB_flag)))
        return {boost::none};
    } catch (std::exception &e) {
      return {boost::none};
    }

    auto const uid = bh->getBoundaryUID(UBB_flag);
    auto const &ubb = bh->template getBoundaryCondition<UBB>(uid);
    return {to_vector3d(
        ubb.getForce((*bc).cell.x(), (*bc).cell.y(), (*bc).cell.z()))};
  };
  bool remove_node_from_boundary(const Utils::Vector3i &node) override {
    auto bc = get_block_and_cell(node, true, m_blocks, n_ghost_layers());
    if (!bc)
      return false;
    Boundaries *boundary_handling =
        (*bc).block->template getData<Boundaries>(m_boundary_handling_id);
    boundary_handling->removeBoundary((*bc).cell[0], (*bc).cell[1],
                                      (*bc).cell[2]);
    return true;
  };
  boost::optional<bool>
  get_node_is_boundary(const Utils::Vector3i &node,
                       bool consider_ghosts = false) const override {
    auto bc =
        get_block_and_cell(node, consider_ghosts, m_blocks, n_ghost_layers());
    if (!bc)
      return {boost::none};

    Boundaries *boundary_handling =
        (*bc).block->template getData<Boundaries>(m_boundary_handling_id);
    return {boundary_handling->isBoundary((*bc).cell)};
  };
  void clear_boundaries() override {
    const CellInterval &domain_bb_in_global_cell_coordinates =
        m_blocks->getCellBBFromAABB(
            m_blocks->begin()->getAABB().getExtended(real_c(n_ghost_layers())));
    for (auto block = m_blocks->begin(); block != m_blocks->end(); ++block) {

      Boundaries *boundary_handling =
          block->template getData<Boundaries>(m_boundary_handling_id);

      CellInterval domain_bb(domain_bb_in_global_cell_coordinates);
      m_blocks->transformGlobalToBlockLocalCellInterval(domain_bb, *block);

      boundary_handling->fillWithDomain(domain_bb);
    }
  };

  // Pressure tensor
  boost::optional<Utils::Vector6d>
  get_node_pressure_tensor(const Utils::Vector3i &node) const override {
    auto bc = get_block_and_cell(node, false, m_blocks, n_ghost_layers());
    if (!bc)
      return {boost::none};
    auto pdf_field = (*bc).block->template getData<PdfField>(m_pdf_field_id);
    return to_vector6d(pdf_field->getPressureTensor((*bc).cell));
  };

  // Global momentum
  Utils::Vector3d get_momentum() const override {
    Vector3<real_t> mom;
    for (auto block_it = m_blocks->begin(); block_it != m_blocks->end();
         ++block_it) {
      auto pdf_field = block_it->template getData<PdfField>(m_pdf_field_id);
      Vector3<real_t> local_v;
      WALBERLA_FOR_ALL_CELLS_XYZ(pdf_field, {
        real_t local_dens = pdf_field->getDensityAndVelocity(local_v, x, y, z);
        mom += local_dens * local_v;
      });
    }
    return to_vector3d(mom);
  };
  // Global external force
  void set_external_force(const Utils::Vector3d &ext_force) override {
    m_reset_force->set_ext_force(ext_force);
  };
  Utils::Vector3d get_external_force() const override {
    return m_reset_force->get_ext_force();
  };

  double get_kT() const override { return 0.; };

  // Grid, domain, halo
  int n_ghost_layers() const override { return m_n_ghost_layers; };
  Utils::Vector3i get_grid_dimensions() const override {
    return m_grid_dimensions;
  }
  std::pair<Utils::Vector3d, Utils::Vector3d>
  get_local_domain() const override {
    // We only have one block per mpi rank
    assert(++(m_blocks->begin()) == m_blocks->end());

    auto const ab = m_blocks->begin()->getAABB();
    return {to_vector3d(ab.min()), to_vector3d(ab.max())};
  };

  bool node_in_local_domain(const Utils::Vector3i &node) const override {
    // Note: Lattice constant =1, cell centers offset by .5
    return get_block_and_cell(node, false, m_blocks, n_ghost_layers()) !=
           boost::none;
  };
  bool node_in_local_halo(const Utils::Vector3i &node) const override {
    return get_block_and_cell(node, true, m_blocks, n_ghost_layers()) !=
           boost::none;
  };
  bool pos_in_local_domain(const Utils::Vector3d &pos) const override {
    return get_block(pos, false, m_blocks, n_ghost_layers()) != nullptr;
  };
  bool pos_in_local_halo(const Utils::Vector3d &pos) const override {
    return get_block(pos, true, m_blocks, n_ghost_layers()) != nullptr;
  };

  std::vector<std::pair<Utils::Vector3i, Utils::Vector3d>>
  node_indices_positions(bool include_ghosts = false) const override {
    int ghost_offset = 0;
    if (include_ghosts)
      ghost_offset = m_n_ghost_layers;
    std::vector<std::pair<Utils::Vector3i, Utils::Vector3d>> res;
    for (auto block = m_blocks->begin(); block != m_blocks->end(); ++block) {
      auto left = block->getAABB().min();
      // Lattice constant is 1, node centers are offset by .5
      Utils::Vector3d pos_offset =
          to_vector3d(left) + Utils::Vector3d::broadcast(.5);

      // Lattice constant is 1, so cast left corner position to ints
      Utils::Vector3i index_offset =
          Utils::Vector3i{int(left[0]), int(left[1]), int(left[2])};

      // Get field data which knows about the indices
      // In the loop, x,y,z are in block-local coordinates
      auto pdf_field = block->template getData<PdfField>(m_pdf_field_id);
      for (int x = -ghost_offset; x < int(pdf_field->xSize()) + ghost_offset;
           x++) {
        for (int y = -ghost_offset; y < int(pdf_field->ySize()) + ghost_offset;
             y++) {
          for (int z = -ghost_offset;
               z < int(pdf_field->zSize()) + ghost_offset; z++) {
            res.push_back({index_offset + Utils::Vector3i{x, y, z},
                           pos_offset + Utils::Vector3d{double(x), double(y),
                                                        double(z)}});
          }
        }
      }
    }
    return res;
  };

  void create_vtk(unsigned delta_N, unsigned initial_count,
                  unsigned flag_observables, std::string const &identifier,
                  std::string const &base_folder,
                  std::string const &prefix) override {
    // VTKOuput object must be unique
    std::stringstream unique_identifier;
    unique_identifier << base_folder << "/" << identifier;
    std::string const vtk_uid = unique_identifier.str();
    if (m_vtk_auto.find(vtk_uid) != m_vtk_auto.end() or
        m_vtk_manual.find(vtk_uid) != m_vtk_manual.end()) {
      throw std::runtime_error("VTKOutput object " + vtk_uid +
                               " already exists");
    }

    // instantiate VTKOutput object
    unsigned const write_freq = (delta_N) ? static_cast<unsigned>(delta_N) : 1u;
    auto pdf_field_vtk = vtk::createVTKOutput_BlockData(
        m_blocks, identifier, uint_c(write_freq), uint_c(0), false, base_folder,
        prefix, true, true, true, true, uint_c(initial_count));
    field::FlagFieldCellFilter<FlagField> fluid_filter(m_flag_field_id);
    fluid_filter.addFlag(Fluid_flag);
    pdf_field_vtk->addCellInclusionFilter(fluid_filter);

    // add writers
    if (static_cast<unsigned>(OutputVTK::density) & flag_observables) {
      pdf_field_vtk->addCellDataWriter(
          make_shared<lbm::DensityVTKWriter<LatticeModel, float>>(
              m_pdf_field_id, "DensityFromPDF"));
    }
    if (static_cast<unsigned>(OutputVTK::velocity_vector) & flag_observables) {
      pdf_field_vtk->addCellDataWriter(
          make_shared<lbm::VelocityVTKWriter<LatticeModel, float>>(
              m_pdf_field_id, "VelocityFromPDF"));
    }
    if (static_cast<unsigned>(OutputVTK::pressure_tensor) & flag_observables) {
      pdf_field_vtk->addCellDataWriter(
          make_shared<lbm::PressureTensorVTKWriter<LatticeModel, float>>(
              m_pdf_field_id, "PressureTensorFromPDF"));
    }

    // register object
    if (delta_N) {
      m_vtk_auto[vtk_uid] = {pdf_field_vtk, true};
    } else {
      m_vtk_manual[vtk_uid] = pdf_field_vtk;
    }
  }

  /** Manually call a VTK callback */
  void write_vtk(std::string const &vtk_uid) override {
    if (m_vtk_auto.find(vtk_uid) != m_vtk_auto.end()) {
      throw std::runtime_error("VTKOutput object " + vtk_uid +
                               " is an automatic observable");
    }
    if (m_vtk_manual.find(vtk_uid) == m_vtk_manual.end()) {
      throw std::runtime_error("VTKOutput object " + vtk_uid +
                               " doesn't exist");
    }
    vtk::writeFiles(m_vtk_manual[vtk_uid])();
  }

  /** Activate or deactivate a VTK callback */
  void switch_vtk(std::string const &vtk_uid, int status) override {
    if (m_vtk_manual.find(vtk_uid) != m_vtk_manual.end()) {
      throw std::runtime_error("VTKOutput object " + vtk_uid +
                               " is a manual observable");
    }
    if (m_vtk_auto.find(vtk_uid) == m_vtk_auto.end()) {
      throw std::runtime_error("VTKOutput object " + vtk_uid +
                               " doesn't exist");
    }
    m_vtk_auto[vtk_uid].second = status;
  }

  /** @brief call, if the lattice model was changed */
  void on_lattice_model_change() {
    for (auto b = m_blocks->begin(); b != m_blocks->end(); ++b) {
      auto pdf_field = b->template getData<PdfField>(m_pdf_field_id);
      pdf_field->resetLatticeModel(*m_lattice_model);
      pdf_field->latticeModel().configure(*b, *m_blocks);
    }
  }

  // pe utility functions
  pe::BodyID get_particle(std::uint64_t uid) const {
    auto it = m_pe_particles.find(uid);
    if (it != m_pe_particles.end()) {
      return it->second;
    }
    return nullptr;
  }
  // pe interface functions
  bool is_particle_on_this_process(std::uint64_t uid) const override {
    pe::BodyID p = get_particle(uid);
    return p != nullptr;
  }
  bool add_particle(std::uint64_t uid, Utils::Vector3d const &gpos,
                    double radius, Utils::Vector3d const &linVel,
                    std::string const &material_name = "iron") override {
    if (m_pe_particles.find(uid) != m_pe_particles.end()) {
      return false;
    }
    auto material = pe::Material::find(material_name);
    if (material == pe::invalid_material)
      material = pe::Material::find("iron");
    pe::SphereID sp = pe::createSphere(
        *m_global_body_storage, m_blocks->getBlockStorage(), m_body_storage_id,
        uid, to_vector3(gpos), real_c(radius), material);
    if (sp != nullptr) {
      sp->setLinearVel(to_vector3(linVel));
      m_pe_particles[uid] = sp;
      m_particle_forces[uid] = sp->getForce();
      m_particle_torques[uid] = sp->getTorque();
      return true;
    }
    return false;
  }
  /** @brief removes all rigid bodies matching the given uid */
  void remove_particle(std::uint64_t uid) override {
    auto it = m_pe_particles.find(uid);
    if (it != m_pe_particles.end()) {
      m_pe_particles.erase(it);
    }
    // Has to be called the same way on all processes to work correctly!
    // That's why it is not inside the if statement above.
    pe::destroyBodyByUID(*m_global_body_storage, m_blocks->getBlockStorage(),
                         m_body_storage_id, uid);
  }
  /** @brief Call after all pe particles have been added to sync them on all
   * blocks */
  void sync_particles() override {
    if (!is_pe_initialized()) {
      throw std::runtime_error("PE is not initialized");
    }
    m_pe_sync_call();
  }
  boost::optional<Utils::Vector3d>
  get_particle_velocity(std::uint64_t uid) const override {
    pe::BodyID p = get_particle(uid);
    if (p != nullptr) {
      return {to_vector3d(p->getLinearVel())};
    }
    return {};
  }
  boost::optional<Utils::Vector3d>
  get_particle_angular_velocity(std::uint64_t uid) const override {
    pe::BodyID p = get_particle(uid);
    if (p != nullptr) {
      return {to_vector3d(p->getAngularVel())};
    }
    return {};
  }
  boost::optional<Utils::Quaternion<double>>
  get_particle_orientation(std::uint64_t uid) const override {
    pe::BodyID p = get_particle(uid);
    if (p != nullptr) {
      return {to_quaternion<real_t>(p->getQuaternion())};
    }
    return {};
  }
  boost::optional<Utils::Vector3d>
  get_particle_position(std::uint64_t uid) const override {
    pe::BodyID p = get_particle(uid);
    if (p != nullptr) {
      return {to_vector3d(p->getPosition())};
    }
    return {};
  }
  boost::optional<Utils::Vector3d>
  get_particle_force(std::uint64_t uid) const override {
    pe::BodyID p = get_particle(uid);
    if (p != nullptr) {
      return {to_vector3d(m_particle_forces.at(uid))};
    }
    return {};
  }
  boost::optional<Utils::Vector3d>
  get_particle_torque(std::uint64_t uid) const override {
    pe::BodyID p = get_particle(uid);
    if (p != nullptr) {
      return {to_vector3d(m_particle_torques.at(uid))};
    }
    return {};
  }

  bool set_particle_force(std::uint64_t uid,
                          Utils::Vector3d const &f) override {
    pe::BodyID p = get_particle(uid);
    if (p == nullptr)
      return false;

    p->setForce(to_vector3(f));
    return true;
  }
  bool add_particle_force(std::uint64_t uid,
                          Utils::Vector3d const &f) override {
    pe::BodyID p = get_particle(uid);
    if (p == nullptr)
      return false;

    p->addForce(to_vector3(f));
    return true;
  }
  bool set_particle_torque(std::uint64_t uid,
                           Utils::Vector3d const &tau) override {
    pe::BodyID p = get_particle(uid);
    if (p == nullptr)
      return false;

    p->setTorque(to_vector3(tau));
    return true;
  }
  bool add_particle_torque(std::uint64_t uid,
                           Utils::Vector3d const &tau) override {
    pe::BodyID p = get_particle(uid);
    if (p == nullptr)
      return false;

    p->addTorque(to_vector3(tau));
    return true;
  }

  void create_particle_material(std::string const &name, double density,
                                double cor, double csf, double cdf,
                                double poisson, double young, double stiffness,
                                double dampingN, double dampingT) override {

    pe::createMaterial(name, real_c(density), real_c(cor), real_c(csf),
                       real_c(cdf), real_c(poisson), real_c(young),
                       real_c(stiffness), real_c(dampingN), real_c(dampingT));
  }

  void map_particles_to_lb_grid() override {
    pe_coupling::mapMovingBodies<Boundaries>(
        *m_blocks, m_boundary_handling_id, m_body_storage_id,
        *m_global_body_storage, m_body_field_id, MO_BB_Flag,
        pe_coupling::selectRegularBodies);
  }
  void finish_particle_adding() override {
    if (!is_pe_initialized()) {
      throw std::runtime_error("PE is not initialized");
    }
    sync_particles();
    map_particles_to_lb_grid();
    if (m_pe_parameters.average_force_torque_over_two_timesteps) {
      m_bodies_force_torque_container_2->store();
    }
  }

  ~LBWalberlaImpl() override = default;
};
} // namespace walberla

#endif // LB_WALBERLA_H