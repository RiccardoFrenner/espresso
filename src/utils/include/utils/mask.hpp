#ifndef ESPRESSO_MASK_HPP
#define ESPRESSO_MASK_HPP

#include <utils/Array.hpp>
#include <utils/get.hpp>

#include <type_traits>
#include <utility>

namespace Utils {
namespace detail {
template <class T, class Integral, size_t... I>
auto mask_impl(Integral mask, T t, std::index_sequence<I...>) {
  return T{((mask & (1u << I)) ? get<I>(t) : tuple_element_t<I, T>{})...};
}
} // namespace detail

/**
 * @brief Pick elements of a tuple-like by a bit mask.
 *
 * E.g. every element of the input for which the corresponding
 * bit is set in the mask is set is copied to the output unmodified,
 * the elements that are not set are set to zero (default constructed
 * instance of the type).
 *
 * Example:
 *   mask(0b1011, {1, 2, 3, 4}) => {1, 0, 3, 4}
 *
 * @tparam T implements the tuple interface(get, tuple_size, ...)
 * @tparam Integral An unsigned integral type
 * @param mask bit mask, if the i-th bit is set, the i-th element
 *        in @param t is copied to the output, otherwise it is set to zero.
 * @param t input elements
 * @return t partially zeroed out according to mask
 */
template <class T, class Integral>
auto mask(Integral mask, T t)
    -> std::enable_if_t<std::is_unsigned<Integral>::value &&
                            (8 * sizeof(Integral) >= tuple_size<T>::value),
                        T> {
  return detail::mask_impl(mask, t,
                           std::make_index_sequence<tuple_size<T>::value>{});
}
} // namespace Utils

#endif // ESPRESSO_MASK_HPP
