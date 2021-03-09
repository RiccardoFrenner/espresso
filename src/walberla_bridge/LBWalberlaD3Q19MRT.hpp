#include "LBWalberlaImpl.hpp"
#include "generated_kernels/MRTLatticeModel.h"

namespace walberla {
class LBWalberlaD3Q19MRT : public LBWalberlaImpl<lbm::MRTLatticeModel> {
public:
  using LatticeModel = lbm::MRTLatticeModel;

  void construct_lattice_model(double viscosity) {
    const real_t omega = 2 / (6 * real_c(viscosity) + 1);
    const real_t magic_number = real_c(3.) / real_c(16.);
    const real_t omega_2 =
        (4 - 2 * omega) / (4 * magic_number * omega + 2 - omega);
    m_lattice_model = std::make_shared<LatticeModel>(
        LatticeModel(m_last_applied_force_field_id,
                     omega,   // bulk
                     omega,   // even
                     omega_2, // odd
                     omega)); // shear
  };
  void set_viscosity(double viscosity) override {
    LatticeModel *lm = dynamic_cast<LatticeModel *>(m_lattice_model.get());
    const real_t omega = 2 / (6 * real_c(viscosity) + 1);
    const real_t magic_number = real_c(3.) / real_c(16.);
    const real_t omega_2 =
        (4 - 2 * omega) / (4 * magic_number * omega + 2 - omega);
    lm->omega_shear_ = omega;
    lm->omega_odd_ = omega_2;
    lm->omega_even_ = omega;
    lm->omega_bulk_ = omega;
    on_lattice_model_change();
  };
  double get_viscosity() const override {
    LatticeModel *lm = dynamic_cast<LatticeModel *>(m_lattice_model.get());
    return (2 - lm->omega_shear_) / (6 * lm->omega_shear_);
  };

  LBWalberlaD3Q19MRT(double viscosity, double density, double tau,
                     const Utils::Vector3i &n_blocks,
                     const Utils::Vector3i &n_cells_per_block,
                     const double lb_cell_size,
                     const Utils::Vector3i &n_processes, int n_ghost_layers,
                     const PE_Parameters &peParams = PE_Parameters())
      : LBWalberlaImpl(n_blocks, n_cells_per_block, lb_cell_size, n_processes,
                       n_ghost_layers, peParams) {
    construct_lattice_model(viscosity);
    setup_with_valid_lattice_model(density);
  };

  LBWalberlaD3Q19MRT(double viscosity, double density, double agrid, double tau,
                     const Utils::Vector3d &box_dimensions,
                     const Utils::Vector3i &node_grid, int n_ghost_layers,
                     const PE_Parameters &peParams = PE_Parameters())
      : LBWalberlaImpl(viscosity, agrid, tau, box_dimensions, node_grid,
                       n_ghost_layers, peParams) {
    construct_lattice_model(viscosity);
    setup_with_valid_lattice_model(density);
  };
};

} // namespace walberla
