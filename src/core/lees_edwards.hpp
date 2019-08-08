#ifndef LEES_EDWARDS_H
#define LEES_EDWARDS_H

#include "cells.hpp"
#include "config.hpp"

/** \file lees_edwards.hpp
 *
 */

class LeesEdwardsProtocol {
public:
  virtual double velocity(double time) const = 0;
  virtual double offset(double time) const = 0;
};

class LeesEdwardsProtocolConstShearVelocity : LeesEdwardsProtocol {
public:
  double velocity(double time) const override { return shear_velocity; };
  double offset(double time) const override {
    return (time - time0) * shear_velocity;
  }
  double shear_velocity;
  double time0;
};

class LeesEdwardsProtocolConstOffset : LeesEdwardsProtocol {
public:
  double velocity(double time) const override { return 0.; };
  double offset(double time) const override { return step; };
  double step;
  double time0;
};

class LeesEdwardsProtocolOscillatoryShear : LeesEdwardsProtocol {
public:
  double velocity(double time) const override {
    return amplitude * std::sin(frequency * (time - time0));
  }
  double offset(double time) const override {
    return frequency * amplitude * std::cos(frequency * (time - time0));
  }
  double amplitude;
  double frequency;
  double time0;
};

/** Enum for the different Lees Edwards Protocols: Off, Step. Steady Shear and
 * Oscillatory shear  */
enum LeesEdwardsProtocolType {
  LEES_EDWARDS_PROTOCOL_OFF,
  LEES_EDWARDS_PROTOCOL_STEP,
  LEES_EDWARDS_PROTOCOL_STEADY_SHEAR,
  LEES_EDWARDS_PROTOCOL_OSC_SHEAR,
};

/** Struct holding all information concerning Lees Edwards  */
typedef struct {
  /** Protocol type*/
  int type;
  /** Time when Lees Edwards was started*/
  double time0;
  /** Current offset*/
  double offset;
  /** Current velocity*/
  double velocity;
  /** Amplitude set via interface*/
  double amplitude;
  /** Frequency set via interface*/
  double frequency;
  /** Direction in which the velocity and position jump is applied*/
  int sheardir;
  /** Direction in which get_mi_vector needs to be modified*/
  int shearplanenormal;
} lees_edwards_protocol_struct;

bool less_edwards_supports_verlet_list();
/** Checks if Lees Edwards supports Verlet lists*/
extern lees_edwards_protocol_struct lees_edwards_protocol;
/** Calculation of current offset*/
double lees_edwards_get_offset(double time);
/** Calculation of current velocity*/
double lees_edwards_get_velocity(double time);
/** At the beginning of Lees_Edwards we have to reset reset all particle images
 * to zero*/
void local_lees_edwards_image_reset();
#endif