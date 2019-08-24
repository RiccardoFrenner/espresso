#ifndef CORE_BOX_GEOMETRY_HPP
#define CORE_BOX_GEOMETRY_HPP

#include <utils/Vector.hpp>

#include <bitset>
#include <cassert>
#include <memory>

#include "config.hpp"
#ifdef LEES_EDWARDS
#include "lees_edwards.hpp"
#endif

class BoxGeometry {
public:
  /** Flags for all three dimensions whether pbc are applied (default). */
  std::bitset<3> m_periodic = 0b111;
  /** Side lengths of the box */
  Utils::Vector3d m_length = {1, 1, 1};
  Utils::Vector3d m_length_half = {.5,.5,.5};
  Utils::Vector3d m_length_inv = {1, 1, 1};

  /**
   * @brief Set periodicity for direction
   *
   * @param coord The coordinate to set the periodicity for.
   * @param val True if this direction should be periodic.
   */
  void set_periodic(unsigned coord, bool val) { m_periodic.set(coord, val); }

  /**
   * @brief Check periodicity in direction.
   *
   * @param coord Direction to check
   * @return true iff periodic in direction.
   */
  constexpr bool periodic(unsigned coord) const {
    assert(coord <= 2);
    return m_periodic[coord];
  }

  /**
   * @brief Box length
   * @return Return vector of side-lengths of the box.
   */
  Utils::Vector3d const &length() const { return m_length; }
  Utils::Vector3d const &length_half() const { return m_length_half; }
  Utils::Vector3d const &length_inv() const { return m_length_inv; }

  /**
   * @brief Set box side lengths.
   * @param box_l Length that should be set.
   */
  void set_length(Utils::Vector3d const &box_l) { 
    m_length = box_l;
    m_length_half = 0.5 * box_l;
    m_length_inv = {1./box_l[0],1./box_l[1],1./box_l[2]};

  }

#ifdef LEES_EDWARDS
  LeesEdwards::Cache lees_edwards_state;
  std::shared_ptr<LeesEdwards::ActiveProtocol> lees_edwards_protocol;
#endif
};

#endif // CORE_BOX_GEOMETRY_HPP
