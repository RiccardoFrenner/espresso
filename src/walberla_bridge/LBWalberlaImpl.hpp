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
#include "ResetForce.hpp"
#include "walberla_utils.hpp"

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
const FlagUID
    MO_BB_Flag("moving obstacle BB"); // TODO: Needed? Same as UBB_flag?
const FlagUID FormerMO_Flag("former moving obstacle");

/** Class that runs and controls the LB on WaLBerla
 */
template <typename LatticeModel> class LBWalberlaImpl : public LBWalberlaBase {
  // TODO: Gibt es einen Grund dass die eigentlich protected sind?
  // public:
  //   using PdfField = lbm::PdfField<LatticeModel>;
  //   using Boundaries =
  //       BoundaryHandling<FlagField, typename LatticeModel::Stencil, UBB,
  //       MO_BB_T>;

protected:
  // Type definitions
  using VectorField = GhostLayerField<real_t, 3>;
  using FlagField = walberla::FlagField<walberla::uint8_t>;
  using PdfField = lbm::PdfField<LatticeModel>;
  using BodyField_T = GhostLayerField<pe::BodyID, 1>;
  /** Velocity boundary conditions */
  using UBB = lbm::UBB<LatticeModel, uint8_t, true, true>;
  typedef pe_coupling::SimpleBB<LatticeModel, FlagField> MO_BB_T;

  /** Boundary handling */
  using Boundaries =
      BoundaryHandling<FlagField, typename LatticeModel::Stencil, UBB>;
  using MOBoundaries =
      BoundaryHandling<FlagField, typename LatticeModel::Stencil, UBB, MO_BB_T>;

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
  // Global particle storage (for very large particles), stored on all processes
  std::shared_ptr<pe::BodyStorage> m_globalBodyStorage;

  // Storage handle for pe particles
  BlockDataID m_body_storage_id;

  // coarse and fine collision detection
  BlockDataID m_ccdID;
  BlockDataID m_fcdID;

  // pe time integrator
  std::shared_ptr<pe::cr::HCSITS> m_cr;

  // storage for force/torque to average over two timesteps
  std::shared_ptr<pe_coupling::BodiesForceTorqueContainer> bodiesFTContainer1;
  std::shared_ptr<pe_coupling::BodiesForceTorqueContainer> bodiesFTContainer2;

  // calculates particle information on call
  // std::shared_ptr<pe::BodyStatistics> m_bodyStats;

  // view on particles to access them via their uid
  std::map<std::uint64_t, pe::BodyID> m_pe_particles;

  size_t stencil_size() const override {
    return static_cast<size_t>(LatticeModel::Stencil::Size);
  }

  // Boundary handling
  class LBBoundaryHandling {
  public:
    LBBoundaryHandling(const BlockDataID &flag_field_id,
                       const BlockDataID &pdf_field_id)
        : m_flag_field_id(flag_field_id), m_pdf_field_id(pdf_field_id) {}

    Boundaries *operator()(IBlock *const block) {

      FlagField *flag_field =
          block->template getData<FlagField>(m_flag_field_id);
      PdfField *pdf_field = block->template getData<PdfField>(m_pdf_field_id);

      const auto fluid = flag_field->flagExists(Fluid_flag)
                             ? flag_field->getFlag(Fluid_flag)
                             : flag_field->registerFlag(Fluid_flag);

      return new Boundaries(
          "boundary handling", flag_field, fluid,
          UBB("velocity bounce back", UBB_flag, pdf_field, nullptr));
    }

  private:
    const BlockDataID m_flag_field_id;
    const BlockDataID m_pdf_field_id;
  };

  // Boundary handling
  class MOBoundaryHandling {
  public:
    MOBoundaryHandling(const BlockDataID &flag_field_id,
                       const BlockDataID &pdf_field_id,
                       const BlockDataID &body_field_id)
        : m_flag_field_id(flag_field_id), m_pdf_field_id(pdf_field_id),
          m_body_field_id(body_field_id) {}

    Boundaries *operator()(IBlock *const block,
                           const StructuredBlockStorage *const storage) const {

      WALBERLA_ASSERT_NOT_NULLPTR(block);
      WALBERLA_ASSERT_NOT_NULLPTR(storage);
      FlagField *flag_field =
          block->template getData<FlagField>(m_flag_field_id);
      PdfField *pdf_field = block->template getData<PdfField>(m_pdf_field_id);
      BodyField_T *body_field =
          block->template getData<BodyField_T>(m_body_field_id);

      const auto fluid = flag_field->flagExists(Fluid_flag)
                             ? flag_field->getFlag(Fluid_flag)
                             : flag_field->registerFlag(Fluid_flag);

      Boundaries *handling = new Boundaries(
          "moving obstacle boundary handling", flag_field, fluid,
          UBB("velocity bounce back", UBB_flag, pdf_field, nullptr),
          MO_BB_T("MO_BB", MO_BB_Flag, pdf_field, flag_field, body_field, fluid,
                  *storage, *block));

      // boundary conditions are set by mapping the planes into the domain
      // TODO: needed?
      // handling->fillWithDomain(FieldGhostLayers);
      return handling;
    }

  private:
    const BlockDataID m_flag_field_id;
    const BlockDataID m_pdf_field_id;
    const BlockDataID m_body_field_id;
  };

public:
  LBWalberlaImpl(double viscosity, double agrid, double tau,
                 const Utils::Vector3d &box_dimensions,
                 const Utils::Vector3i &node_grid, int n_ghost_layers) {
    m_n_ghost_layers = n_ghost_layers;

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

    m_blocks = blockforest::createUniformBlockGrid(
        uint_c(node_grid[0]), // blocks in x direction
        uint_c(node_grid[1]), // blocks in y direction
        uint_c(node_grid[2]), // blocks in z direction
        uint_c(m_grid_dimensions[0] /
               node_grid[0]), // number of cells per block in x direction
        uint_c(m_grid_dimensions[1] /
               node_grid[1]), // number of cells per block in y direction
        uint_c(m_grid_dimensions[2] /
               node_grid[2]), // number of cells per block in z direction
        1,                    // Lattice constant
        uint_c(node_grid[0]), uint_c(node_grid[1]),
        uint_c(node_grid[2]), // cpus per direction
        true, true, true);

    // Init and register force fields
    m_last_applied_force_field_id = field::addToStorage<VectorField>(
        m_blocks, "force field", real_t{0}, field::fzyx, m_n_ghost_layers);
    m_force_to_be_applied_id = field::addToStorage<VectorField>(
        m_blocks, "force field", real_t{0}, field::fzyx, m_n_ghost_layers);

    // Init and register flag field (fluid/boundary)
    m_flag_field_id = field::addFlagFieldToStorage<FlagField>(
        m_blocks, "flag field", m_n_ghost_layers);

    // Init pe
    // Add body field
    m_body_field_id = field::addToStorage<BodyField_T>(
        m_blocks, "body field", nullptr, field::zyxf, m_n_ghost_layers);

    m_globalBodyStorage = std::make_shared<pe::BodyStorage>();

    // Storage for pe particles
    m_body_storage_id = m_blocks->addBlockData(
        pe::createStorageDataHandling<BodyTypeTuple>(), "PE Body Storage");

    // coarse and fine collision detection
    m_ccdID =
        m_blocks->addBlockData(pe::ccd::createHashGridsDataHandling(
                                   m_globalBodyStorage, m_body_storage_id),
                               "CCD");
    m_fcdID = m_blocks->addBlockData(
        pe::fcd::createGenericFCDDataHandling<
            BodyTypeTuple, pe::fcd::AnalyticCollideFunctor>(),
        "FCD");

    // Init pe time integrator
    // cr::HCSITS is for hard contacts, cr::DEM for soft contacts
    // TODO: Access to m_cr parameters
    m_cr = std::make_shared<pe::cr::HCSITS>(
        m_globalBodyStorage, m_blocks->getBlockStoragePointer(),
        m_body_storage_id, m_ccdID, m_fcdID, nullptr);
    m_cr->setMaxIterations(10);
    m_cr->setRelaxationModel(
        pe::cr::HardContactSemiImplicitTimesteppingSolvers::
            ApproximateInelasticCoulombContactByDecoupling);
    m_cr->setRelaxationParameter(real_t(0.7));
    m_cr->setGlobalLinearAcceleration(pe::Vec3(0, 0, 0));

    // Init pe body type ids
    pe::SetBodyTypeIDs<BodyTypeTuple>::execute();

    // Init body statistics
    // m_bodyStats =
    //     std::make_shared<pe::BodyStatistics>(m_blocks, m_body_storage_id);
    // (*m_bodyStats)();

    // Init body force/torque storage
    bodiesFTContainer1 =
        std::make_shared<pe_coupling::BodiesForceTorqueContainer>(
            m_blocks, m_body_storage_id);
    bodiesFTContainer2 =
        std::make_shared<pe_coupling::BodiesForceTorqueContainer>(
            m_blocks, m_body_storage_id);
  };

  void setup_with_valid_lattice_model(double density) {
    // Init and register pdf field
    m_pdf_field_id = lbm::addPdfFieldToStorage(
        m_blocks, "pdf field", *(m_lattice_model.get()),
        to_vector3(Utils::Vector3d{}), real_t(density), m_n_ghost_layers);

    // Register boundary handling
    m_boundary_handling_id = m_blocks->addStructuredBlockData<Boundaries>(
        LBBoundaryHandling(m_flag_field_id, m_pdf_field_id, m_body_field_id),
        "boundary handling");
    clear_boundaries();

    // map pe bodies into the LBM simulation
    // uses standard bounce back boundary conditions
    pe_coupling::mapMovingBodies<
        Boundaries>(/* TODO: Das sollte erst nach dem Hinzufügen von Partikeln
                       gecallt werden! Allgemein sollte mein ganzes pe Zeug
                       nicht unbedingt in dieser Funktion stehen */
                    *m_blocks, m_boundary_handling_id, m_body_storage_id,
                    *m_globalBodyStorage, m_body_field_id, MO_BB_Flag,
                    pe_coupling::selectRegularBodies);

    // sets up the communication and registers pdf field and force field to it
    // TODO: m_communication = scheme in SegreSilberbergMem.cpp
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

    // Add steps to the integration loop
    m_time_loop = std::make_shared<timeloop::SweepTimeloop>(
        m_blocks->getBlockStorage(), 1);

    // sweep for updating the pe body mapping into the LBM simulation
    m_time_loop->add() << timeloop::Sweep(
        pe_coupling::BodyMapping<LatticeModel, Boundaries>(
            m_blocks, m_pdf_field_id, m_boundary_handling_id, m_body_storage_id,
            m_globalBodyStorage, m_body_field_id, MO_BB_Flag, FormerMO_Flag,
            pe_coupling::selectRegularBodies),
        "Body Mapping");

    // sweep for restoring PDFs in cells previously occupied by pe bodies
    // this uses the ´EquilibriumReconstructor´ alternatives are
    // ´EquilibriumAndNonEquilibriumReconstructor´ and
    // ´ExtrapolationReconstructor´
    typedef pe_coupling::EquilibriumReconstructor<LatticeModel, Boundaries>
        Reconstructor_T;
    Reconstructor_T reconstructor(m_blocks, m_boundary_handling_id,
                                  m_body_field_id);
    m_time_loop->add() << timeloop::Sweep(
        pe_coupling::PDFReconstruction<LatticeModel, Boundaries,
                                       Reconstructor_T>(
            m_blocks, m_pdf_field_id, m_boundary_handling_id, m_body_storage_id,
            m_globalBodyStorage, m_body_field_id, reconstructor, FormerMO_Flag,
            Fluid_flag),
        "PDF Restore");

    bodiesFTContainer2->store();

    // TODO: In SegreSilberbergMem.cpp the communication function is added as an
    // BeforeFunction, does that matter?
    m_time_loop->add() << timeloop::Sweep(
        Boundaries::getBlockSweep(m_boundary_handling_id), "boundary handling");
    m_time_loop->add() << timeloop::Sweep(makeSharedSweep(m_reset_force),
                                          "Reset force fields");
    m_time_loop->add() << timeloop::Sweep(
                              typename LatticeModel::Sweep(m_pdf_field_id),
                              "LB stream & collide")
                       << timeloop::AfterFunction(*m_communication,
                                                  "communication");

    // Register velocity access adapter (proxy)
    m_velocity_adaptor_id = field::addFieldAdaptor<VelocityAdaptor>(
        m_blocks, m_pdf_field_id, "velocity adaptor");

    // Synchronize ghost layers
    (*m_communication)();

    // Averaging of force/torque over two time steps to damp oscillations of the
    // interaction force/torque
    std::function<void(void)> storeForceTorqueInCont1 = std::bind(
        &pe_coupling::BodiesForceTorqueContainer::store, bodiesFTContainer1);
    std::function<void(void)> setForceTorqueOnBodiesFromCont2 =
        std::bind(&pe_coupling::BodiesForceTorqueContainer::setOnBodies,
                  bodiesFTContainer2);

    // store force/torque from hydrodynamic interactions in container1
    m_time_loop->addFuncAfterTimeStep(storeForceTorqueInCont1, "Force Storing");

    // set force/torque from previous time step (in container2)
    m_time_loop->addFuncAfterTimeStep(setForceTorqueOnBodiesFromCont2,
                                      "Force setting");

    // average the force/torque by scaling it with factor 1/2
    m_time_loop->addFuncAfterTimeStep(
        pe_coupling::ForceTorqueOnBodiesScaler(m_blocks, m_body_storage_id,
                                               real_t(0.5)),
        "Force averaging");

    // swap containers
    m_time_loop->addFuncAfterTimeStep(
        pe_coupling::BodyContainerSwapper(bodiesFTContainer1,
                                          bodiesFTContainer2),
        "Swap FT container");

    // add pe timesteps
    const uint_t numPeSubcycles = uint_c(1); // TODO: member? getter, setter?
    std::function<void(void)> syncCall =
        std::bind(pe::syncNextNeighbors<BodyTypeTuple>,
                  std::ref(m_blocks->getBlockForest()), m_body_storage_id,
                  static_cast<WcTimingTree *>(nullptr), real_t(1.5),
                  false); // TODO: replace
    m_time_loop->addFuncAfterTimeStep(
        pe_coupling::TimeStep(m_blocks, m_body_storage_id, *m_cr, syncCall,
                              real_t(1), numPeSubcycles),
        "pe Time Step");
  };

  std::shared_ptr<LatticeModel> get_lattice_model() { return m_lattice_model; };

  void integrate() override {
    m_time_loop->singleStep();

    // Handle VTK writers
    // TODO: Wieder einkommentieren
    // for (auto it = m_vtk_auto.begin(); it != m_vtk_auto.end(); ++it) {
    //   if (it->second.second)
    //     vtk::writeFiles(it->second.first)();
    // }
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

  // TODO: Do I really need this to track settling sphere pos/vel?
  void add_func_after_time_step(const std::function<void()> &f,
                                const std::string &id) {
    m_time_loop->addFuncAfterTimeStep(f, id);
  }

  // pe utility functions
  pe::BodyID get_pe_particle(std::uint64_t uid) const {
    auto it = m_pe_particles.find(uid);
    if (it != m_pe_particles.end()) {
      return it->second;
    }
    return nullptr;
  }

  // pe interface functions
  bool add_pe_particle(std::uint64_t uid, const Utils::Vector3d &gpos,
                       double radius, const Utils::Vector3d &linVel,
                       const std::string &material_name = "iron") override {
    if (m_pe_particles.find(uid) != m_pe_particles.end()) {
      return false;
    }
    // TODO: Particle as argument? -> not possible since no particle class in
    // espresso
    auto material = pe::Material::find(material_name);
    if (material == pe::invalid_material)
      material = pe::Material::find("iron");
    pe::SphereID sp = pe::createSphere(
        *m_globalBodyStorage, m_blocks->getBlockStorage(), m_body_storage_id,
        uid, to_vector3(gpos), real_c(radius), material);
    if (sp != nullptr) {
      sp->setLinearVel(to_vector3(linVel));
      m_pe_particles[uid] = sp;
      return true;
    }
    return false;
  }

  /** @brief removes all rigid bodies matching the given uid */
  void remove_pe_particle(std::uint64_t uid) override {
    auto it = m_pe_particles.find(uid);
    if (it != m_pe_particles.end()) {
      m_pe_particles.erase(it);
    }
    // Has to be called the same way on all processes to work correctly!
    // That's why it is not inside the if statement above.
    pe::destroyBodyByUID(*m_globalBodyStorage, m_blocks->getBlockStorage(),
                         m_body_storage_id, uid);
  }

  /** @brief Call after all pe particles have been added to sync them on all
   * blocks */
  void sync_pe_particles() override {
    // overlap of 1.5 * lattice grid spacing is needed for correct mapping
    // see
    // https://www.walberla.net/doxygen/group__pe__coupling.html#reconstruction
    // for more information
    real_t dx = real_t(1); // hardcoded in createUniformBlockGrid above
    real_t overlap = real_t(1.5) * dx;
    // TODO: Difference between syncShadowOwners?
    pe::syncNextNeighbors<BodyTypeTuple>(
        m_blocks->getBlockForest(), m_body_storage_id,
        static_cast<WcTimingTree *>(nullptr), overlap, false);
  }

  boost::optional<Utils::Vector3d>
  get_particle_velocity(std::uint64_t uid) const override {
    pe::BodyID p = get_pe_particle(uid);
    if (p != nullptr) {
      return {to_vector3d(p->getLinearVel())};
    }
    return {};
  }
  boost::optional<Utils::Vector3d>
  get_particle_angular_velocity(std::uint64_t uid) const override {
    pe::BodyID p = get_pe_particle(uid);
    if (p != nullptr) {
      return {to_vector3d(p->getAngularVel())};
    }
    return {};
  }
  // TODO: Somehow use real_t instead of always double
  boost::optional<Utils::Quaternion<double>>
  get_particle_orientation(std::uint64_t uid) const override {
    pe::BodyID p = get_pe_particle(uid);
    if (p != nullptr) {
      return {to_quaternion<real_t>(p->getQuaternion())};
    }
    return {};
  }
  boost::optional<Utils::Vector3d>
  get_particle_position(std::uint64_t uid) const override {
    pe::BodyID p = get_pe_particle(uid);
    if (p != nullptr) {
      return {to_vector3d(p->getPosition())};
    }
    return {};
  }
  void set_particle_force(std::uint64_t uid, const Utils::Vector3d &f) {
    pe::BodyID p = get_pe_particle(uid);
    if (p != nullptr) {
      p->setForce(to_vector3(f));
    }
    // If particle not on the calling rank, nothing happens.
    // TODO: Is this the wanted behaviour?
  }
  void set_particle_torque(std::uint64_t uid, const Utils::Vector3d &tau) {
    pe::BodyID p = get_pe_particle(uid);
    if (p != nullptr) {
      p->setTorque(to_vector3(tau));
    }
    // If particle not on the calling rank, nothing happens.
    // TODO: Is this the wanted behaviour?
  }

  void createMaterial(const std::string &name, double density, double cor,
                      double csf, double cdf, double poisson, double young,
                      double stiffness, double dampingN, double dampingT) {

    pe::createMaterial(name, real_c(density), real_c(cor), real_c(csf),
                       real_c(cdf), real_c(poisson), real_c(young),
                       real_c(stiffness), real_c(dampingN), real_c(dampingT));
  }

  // pe coupling interface functions
  void map_moving_bodies() {
    pe_coupling::mapMovingBodies<Boundaries>(
        *m_blocks, m_boundary_handling_id, m_body_storage_id,
        *m_globalBodyStorage, m_body_field_id, MO_BB_Flag,
        pe_coupling::selectRegularBodies);
  }

  ~LBWalberlaImpl() override = default;
};
} // namespace walberla

#endif // LB_WALBERLA_H

// TODO: I have added everything in SegreSilberbergMEM.cpp up to line 737