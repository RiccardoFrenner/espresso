#ifndef PE_PARAMETERS_H
#define PE_PARAMETERS_H

#include <boost/serialization/access.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include <utils/Vector.hpp>

class PE_Parameters {
private:
  using ForceList = std::vector<std::pair<Utils::Vector3d, std::string>>;

  bool m_use_moving_obstacles; // If false other parameters are obsolete
  ForceList m_constant_global_forces;
  std::uint32_t m_num_pe_sub_cycles;
  bool m_sync_shadow_owners;
  bool m_average_force_torque_over_two_timesteps;
  double m_syncronization_overlap_factor; // pe_sync_overlap =
                                          // m_syncronization_overlap_factor *
                                          // LB cell length

public:
  PE_Parameters(ForceList _constant_global_forces = {},
                std::uint32_t _num_pe_sub_cycles = 1,
                bool _sync_shadow_owners = true,
                bool _average_force_torque_over_two_timesteps = false,
                double _syncronization_overlap_factor = 1.5)
      : m_use_moving_obstacles(true),
        m_constant_global_forces(std::move(_constant_global_forces)),
        m_num_pe_sub_cycles(_num_pe_sub_cycles),
        m_sync_shadow_owners(_sync_shadow_owners),
        m_average_force_torque_over_two_timesteps(
            _average_force_torque_over_two_timesteps),
        m_syncronization_overlap_factor(_syncronization_overlap_factor) {}
  static PE_Parameters deactivated() {
    PE_Parameters p;
    p.m_use_moving_obstacles = false;
    return p;
  }
  bool is_activated() const { return m_use_moving_obstacles; }
  ForceList const &get_constant_global_forces() const {
    return m_constant_global_forces;
  }
  std::uint32_t get_num_pe_sub_cycles() const { return m_num_pe_sub_cycles; }
  bool get_sync_shadow_owners() const { return m_sync_shadow_owners; }
  bool get_average_force_torque_over_two_timesteps() const {
    return m_average_force_torque_over_two_timesteps;
  }
  double get_syncronization_overlap_factor() const {
    return m_syncronization_overlap_factor;
  }

private:
  friend boost::serialization::access;
  template <typename Archive>
  void serialize(Archive &ar, const unsigned int /* version */) {
    ar &m_use_moving_obstacles;
    ar &m_constant_global_forces;
    ar &m_num_pe_sub_cycles;
    ar &m_sync_shadow_owners;
    ar &m_average_force_torque_over_two_timesteps;
    ar &m_syncronization_overlap_factor;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  PE_Parameters const &pe_params) {
    if (pe_params.is_activated()) {
      os << "PE_Parameters({";
      for (auto const &f : pe_params.get_constant_global_forces()) {
        os << "{{" << f.first << "}, " << f.second << "}";
      }
      os << "}, " << pe_params.get_num_pe_sub_cycles() << ", "
         << pe_params.get_sync_shadow_owners() << ", "
         << pe_params.get_average_force_torque_over_two_timesteps() << ", "
         << pe_params.get_syncronization_overlap_factor() << ")";
    } else {
      os << "PE_Parameters::deactivated";
    }
    return os;
  }
};

#endif