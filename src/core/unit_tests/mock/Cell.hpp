#ifndef CORE_UNIT_TESTS_MOCK_CELL_HPP
#define CORE_UNIT_TESTS_MOCK_CELL_HPP

#include <functional>
#include <vector>

namespace Testing {
template <typename Particle> class Cell {
public:
  Cell() : n(0) {}

  std::size_t n;
  std::vector<Particle> part;
};
} // namespace Testing

#endif
