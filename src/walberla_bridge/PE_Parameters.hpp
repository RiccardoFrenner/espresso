#ifndef PE_PARAMETERS_H
#define PE_PARAMETERS_H

#include <boost/serialization/access.hpp>
#include <utils/Vector.hpp>

#include <utility>
#include <vector>

struct PE_Parameters {
  bool use_moving_obstacles = false; // If false other parameters are obsolete
  bool sync_shadow_owners = true;

  // pe_sync_overlap = syncronization_overlap_factor * dx
  // Where dx is the LB cell length
  double syncronization_overlap_factor = double(1.5);

  bool average_force_torque_over_two_timesteps = true;
  std::size_t num_pe_sub_cycles = 1;
  std::vector<std::pair<Utils::Vector3d, std::string>> constant_global_forces =
      {};

  PE_Parameters() = default;

  PE_Parameters(bool _use_moving_obstacles, bool _sync_shadow_owners,
                double _syncronization_overlap_factor,
                bool _average_force_torque_over_two_timesteps,
                std::size_t _num_pe_sub_cycles)
      : use_moving_obstacles(_use_moving_obstacles),
        sync_shadow_owners(_sync_shadow_owners),
        syncronization_overlap_factor(_syncronization_overlap_factor),
        average_force_torque_over_two_timesteps(
            _average_force_torque_over_two_timesteps),
        num_pe_sub_cycles(_num_pe_sub_cycles) {}

  void add_global_constant_force(Utils::Vector3d f, std::string name) {
    constant_global_forces.emplace_back(std::make_pair(f, name));
  }

private:
  friend boost::serialization::access;
  template <typename Archive>
  void serialize(Archive &ar, const unsigned int /* version */) {
    ar &use_moving_obstacles;
  }
};

#endif