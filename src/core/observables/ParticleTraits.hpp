#ifndef OBSERVABLES_PARTICLE_TRAITS
#define OBSERVABLES_PARTICLE_TRAITS

#include "Particle.hpp"
#include "config.hpp"

namespace GenObs {
template <> struct traits<Particle> {
  auto position(Particle const &p) const { return p.r.p; }
  auto velocity(Particle const &p) const { return p.m.v; }
  auto mass(Particle const &p) const {
#ifdef VIRTUAL_SITES
    if (p.p.is_virtual)
      return decltype(p.p.mass){};
#endif
    return p.p.mass;
  }
  auto charge(Particle const &p) const { return p.p.q; }
  auto force(Particle const &p) const {
#ifdef VIRTUAL_SITES
    if (p.p.is_virtual)
      return decltype(p.f.f){};
#endif
    return p.f.f;
  }
  auto dipole_moment(Particle const &p) const {
#if defined(ROTATION) && defined(DIPOLES)
    return p.calc_dip();
#else
    return Utils::Vector3d{};
#endif
  }
};

} // namespace GenObs

#endif
