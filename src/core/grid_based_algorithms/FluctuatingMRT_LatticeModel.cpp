//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\author Martin Bauer <martin.bauer@fau.de>
//======================================================================================================================

#include <cmath>

#include "FluctuatingMRT_LatticeModel.h"
#include "core/DataTypes.h"
#include "core/Macros.h"
#include "lbm/field/PdfField.h"
#include "lbm/sweeps/Streaming.h"

#ifdef _MSC_VER
#pragma warning(disable : 4458)
#endif

#define FUNC_PREFIX

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "philox_rand.h"

using namespace std;

namespace walberla {
namespace lbm {

namespace internal_kernel_streamCollide {
static FUNC_PREFIX void kernel_streamCollide(
    double *RESTRICT const _data_force, double *RESTRICT const _data_pdfs,
    double *RESTRICT _data_pdfs_tmp, int64_t const _size_force_0,
    int64_t const _size_force_1, int64_t const _size_force_2,
    int64_t const _stride_force_0, int64_t const _stride_force_1,
    int64_t const _stride_force_2, int64_t const _stride_force_3,
    int64_t const _stride_pdfs_0, int64_t const _stride_pdfs_1,
    int64_t const _stride_pdfs_2, int64_t const _stride_pdfs_3,
    int64_t const _stride_pdfs_tmp_0, int64_t const _stride_pdfs_tmp_1,
    int64_t const _stride_pdfs_tmp_2, int64_t const _stride_pdfs_tmp_3,
    uint32_t block_offset_0, uint32_t block_offset_1, uint32_t block_offset_2,
    double omega_bulk, double omega_even, double omega_odd, double omega_shear,
    uint32_t seed, double temperature, uint32_t time_step) {
  const double xi_35 = omega_shear + 2.0;
  const double xi_36 = xi_35 * 0.5;
  const double xi_41 = xi_35 * 0.0833333333333333;
  const double xi_46 = xi_35 * 0.166666666666667;
  const double xi_56 = xi_35 * 0.25;
  const double xi_61 = xi_35 * 0.0416666666666667;
  const double xi_88 = 2.4494897427831779;
  const double xi_113 = omega_odd * 0.25;
  const double xi_129 = omega_odd * 0.0833333333333333;
  const double xi_194 = omega_shear * 0.25;
  const double xi_209 = omega_odd * 0.0416666666666667;
  const double xi_211 = omega_odd * 0.125;
  const int64_t rr_0 = 0.0;
  const double xi_118 = rr_0 * 0.166666666666667;
  const double xi_184 = rr_0 * 0.0833333333333333;
  for (int ctr_2 = 1; ctr_2 < _size_force_2 - 1; ctr_2 += 1) {
    double *RESTRICT _data_pdfs_2m1_314 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                          _stride_pdfs_2 + 14 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_21_318 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                         _stride_pdfs_2 + 18 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_34 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 4 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_2m1_311 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                          _stride_pdfs_2 + 11 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_31 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + _stride_pdfs_3;
    double *RESTRICT _data_pdfs_21_315 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                         _stride_pdfs_2 + 15 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_2m1_312 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                          _stride_pdfs_2 + 12 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_2m1_35 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                         _stride_pdfs_2 + 5 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_33 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 3 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_39 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 9 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_32 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 2 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_21_316 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                         _stride_pdfs_2 + 16 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_21_317 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                         _stride_pdfs_2 + 17 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_21_36 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                        _stride_pdfs_2 + 6 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_37 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 7 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_2m1_313 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                          _stride_pdfs_2 + 13 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_310 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 10 * _stride_pdfs_3;
    double *RESTRICT _data_force_20_31 =
        _data_force + _stride_force_2 * ctr_2 + _stride_force_3;
    double *RESTRICT _data_force_20_30 = _data_force + _stride_force_2 * ctr_2;
    double *RESTRICT _data_force_20_32 =
        _data_force + _stride_force_2 * ctr_2 + 2 * _stride_force_3;
    double *RESTRICT _data_pdfs_20_30 = _data_pdfs + _stride_pdfs_2 * ctr_2;
    double *RESTRICT _data_pdfs_20_38 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 8 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_30 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2;
    double *RESTRICT _data_pdfs_tmp_20_31 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_32 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 2 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_33 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 3 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_34 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 4 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_35 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 5 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_36 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 6 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_37 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 7 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_38 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 8 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_39 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 9 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_310 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 10 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_311 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 11 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_312 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 12 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_313 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 13 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_314 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 14 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_315 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 15 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_316 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 16 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_317 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 17 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_tmp_20_318 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 18 * _stride_pdfs_tmp_3;
    for (int ctr_1 = 1; ctr_1 < _size_force_1 - 1; ctr_1 += 1) {
      double *RESTRICT _data_pdfs_2m1_314_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_2m1_314;
      double *RESTRICT _data_pdfs_21_318_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_21_318;
      double *RESTRICT _data_pdfs_20_34_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_34;
      double *RESTRICT _data_pdfs_2m1_311_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_2m1_311;
      double *RESTRICT _data_pdfs_20_31_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_20_31;
      double *RESTRICT _data_pdfs_21_315_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_21_315;
      double *RESTRICT _data_pdfs_2m1_312_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_2m1_312;
      double *RESTRICT _data_pdfs_2m1_35_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_2m1_35;
      double *RESTRICT _data_pdfs_20_33_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_33;
      double *RESTRICT _data_pdfs_20_39_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_20_39;
      double *RESTRICT _data_pdfs_20_32_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_20_32;
      double *RESTRICT _data_pdfs_21_316_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_21_316;
      double *RESTRICT _data_pdfs_21_317_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_21_317;
      double *RESTRICT _data_pdfs_21_36_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_21_36;
      double *RESTRICT _data_pdfs_20_37_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_20_37;
      double *RESTRICT _data_pdfs_2m1_313_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_2m1_313;
      double *RESTRICT _data_pdfs_20_310_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_20_310;
      double *RESTRICT _data_force_20_31_10 =
          _stride_force_1 * ctr_1 + _data_force_20_31;
      double *RESTRICT _data_force_20_30_10 =
          _stride_force_1 * ctr_1 + _data_force_20_30;
      double *RESTRICT _data_force_20_32_10 =
          _stride_force_1 * ctr_1 + _data_force_20_32;
      double *RESTRICT _data_pdfs_20_30_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_30;
      double *RESTRICT _data_pdfs_20_38_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_20_38;
      double *RESTRICT _data_pdfs_tmp_20_30_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_30;
      double *RESTRICT _data_pdfs_tmp_20_31_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_31;
      double *RESTRICT _data_pdfs_tmp_20_32_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_32;
      double *RESTRICT _data_pdfs_tmp_20_33_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_33;
      double *RESTRICT _data_pdfs_tmp_20_34_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_34;
      double *RESTRICT _data_pdfs_tmp_20_35_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_35;
      double *RESTRICT _data_pdfs_tmp_20_36_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_36;
      double *RESTRICT _data_pdfs_tmp_20_37_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_37;
      double *RESTRICT _data_pdfs_tmp_20_38_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_38;
      double *RESTRICT _data_pdfs_tmp_20_39_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_39;
      double *RESTRICT _data_pdfs_tmp_20_310_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_310;
      double *RESTRICT _data_pdfs_tmp_20_311_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_311;
      double *RESTRICT _data_pdfs_tmp_20_312_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_312;
      double *RESTRICT _data_pdfs_tmp_20_313_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_313;
      double *RESTRICT _data_pdfs_tmp_20_314_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_314;
      double *RESTRICT _data_pdfs_tmp_20_315_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_315;
      double *RESTRICT _data_pdfs_tmp_20_316_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_316;
      double *RESTRICT _data_pdfs_tmp_20_317_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_317;
      double *RESTRICT _data_pdfs_tmp_20_318_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_318;
      for (int ctr_0 = 1; ctr_0 < _size_force_0 - 1; ctr_0 += 1) {

        float Dummy_299;
        float Dummy_300;
        float Dummy_301;
        float Dummy_302;
        philox_float4(time_step, block_offset_0 + ctr_0, block_offset_1 + ctr_1,
                      block_offset_2 + ctr_2, 3, seed, Dummy_299, Dummy_300,
                      Dummy_301, Dummy_302);

        float Dummy_295;
        float Dummy_296;
        float Dummy_297;
        float Dummy_298;
        philox_float4(time_step, block_offset_0 + ctr_0, block_offset_1 + ctr_1,
                      block_offset_2 + ctr_2, 2, seed, Dummy_295, Dummy_296,
                      Dummy_297, Dummy_298);

        float Dummy_291;
        float Dummy_292;
        float Dummy_293;
        float Dummy_294;
        philox_float4(time_step, block_offset_0 + ctr_0, block_offset_1 + ctr_1,
                      block_offset_2 + ctr_2, 1, seed, Dummy_291, Dummy_292,
                      Dummy_293, Dummy_294);

        float Dummy_287;
        float Dummy_288;
        float Dummy_289;
        float Dummy_290;
        philox_float4(time_step, block_offset_0 + ctr_0, block_offset_1 + ctr_1,
                      block_offset_2 + ctr_2, 0, seed, Dummy_287, Dummy_288,
                      Dummy_289, Dummy_290);

        const double xi_0 =
            _data_pdfs_21_318_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0] +
            _data_pdfs_2m1_314_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_1 =
            xi_0 + _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_2 = _data_pdfs_20_31_1m1[_stride_pdfs_0 * ctr_0] +
                            _data_pdfs_21_315_1m1[_stride_pdfs_0 * ctr_0] +
                            _data_pdfs_2m1_311_1m1[_stride_pdfs_0 * ctr_0];
        const double xi_3 = _data_pdfs_2m1_312_11[_stride_pdfs_0 * ctr_0] +
                            _data_pdfs_2m1_35_10[_stride_pdfs_0 * ctr_0];
        const double xi_4 =
            _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] +
            _data_pdfs_20_39_11[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_5 = _data_pdfs_20_32_11[_stride_pdfs_0 * ctr_0] +
                            _data_pdfs_21_316_11[_stride_pdfs_0 * ctr_0];
        const double xi_6 =
            _data_pdfs_21_317_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] +
            _data_pdfs_21_36_10[_stride_pdfs_0 * ctr_0];
        const double xi_8 =
            -_data_pdfs_20_39_11[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_9 =
            xi_8 -
            _data_pdfs_20_37_1m1[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_10 =
            -_data_pdfs_21_317_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_11 =
            -_data_pdfs_2m1_313_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_12 =
            -_data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_13 = xi_10 + xi_11 + xi_12;
        const double xi_14 = -_data_pdfs_20_32_11[_stride_pdfs_0 * ctr_0];
        const double xi_15 =
            -_data_pdfs_20_310_11[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_16 = xi_14 + xi_15;
        const double xi_17 = -_data_pdfs_21_316_11[_stride_pdfs_0 * ctr_0];
        const double xi_18 = -_data_pdfs_2m1_312_11[_stride_pdfs_0 * ctr_0];
        const double xi_19 = xi_17 + xi_18;
        const double xi_20 =
            -_data_pdfs_21_318_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_21 = xi_10 + xi_20;
        const double xi_22 = -_data_pdfs_21_315_1m1[_stride_pdfs_0 * ctr_0];
        const double xi_23 = -_data_pdfs_21_36_10[_stride_pdfs_0 * ctr_0];
        const double xi_24 = xi_17 + xi_22 + xi_23 +
                             _data_pdfs_2m1_311_1m1[_stride_pdfs_0 * ctr_0];
        const double xi_40 =
            0.166666666666667 * _data_force_20_31_10[_stride_force_0 * ctr_0];
        const double xi_48 =
            0.166666666666667 * _data_force_20_30_10[_stride_force_0 * ctr_0];
        const double xi_52 =
            0.166666666666667 * _data_force_20_32_10[_stride_force_0 * ctr_0];
        const double xi_55 =
            0.5 * _data_force_20_31_10[_stride_force_0 * ctr_0];
        const double xi_59 =
            0.0833333333333333 * _data_force_20_30_10[_stride_force_0 * ctr_0];
        const double xi_63 =
            0.0833333333333333 * _data_force_20_31_10[_stride_force_0 * ctr_0];
        const double xi_73 =
            0.0833333333333333 * _data_force_20_32_10[_stride_force_0 * ctr_0];
        const double xi_91 = -_data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0];
        const double xi_92 = xi_91 +
                             3.0 * _data_pdfs_21_36_10[_stride_pdfs_0 * ctr_0] +
                             3.0 * _data_pdfs_2m1_35_10[_stride_pdfs_0 * ctr_0];
        const double xi_93 =
            omega_even *
            (xi_92 - 3.0 * _data_pdfs_21_315_1m1[_stride_pdfs_0 * ctr_0] -
             3.0 * _data_pdfs_21_316_11[_stride_pdfs_0 * ctr_0] -
             3.0 * _data_pdfs_2m1_311_1m1[_stride_pdfs_0 * ctr_0] -
             3.0 * _data_pdfs_2m1_312_11[_stride_pdfs_0 * ctr_0] +
             3.0 * _data_pdfs_20_31_1m1[_stride_pdfs_0 * ctr_0] +
             3.0 * _data_pdfs_20_32_11[_stride_pdfs_0 * ctr_0]);
        const double xi_94 =
            2.0 * _data_pdfs_21_315_1m1[_stride_pdfs_0 * ctr_0] +
            2.0 * _data_pdfs_21_316_11[_stride_pdfs_0 * ctr_0] +
            2.0 * _data_pdfs_2m1_311_1m1[_stride_pdfs_0 * ctr_0] +
            2.0 * _data_pdfs_2m1_312_11[_stride_pdfs_0 * ctr_0];
        const double xi_95 =
            xi_94 +
            5.0 * _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] +
            5.0 * _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_96 =
            omega_even *
            (xi_92 + xi_95 -
             2.0 * _data_pdfs_20_31_1m1[_stride_pdfs_0 * ctr_0] -
             2.0 * _data_pdfs_20_32_11[_stride_pdfs_0 * ctr_0] -
             5.0 *
                 _data_pdfs_21_317_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] -
             5.0 *
                 _data_pdfs_21_318_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0] -
             5.0 * _data_pdfs_2m1_313_10[_stride_pdfs_0 * ctr_0 +
                                         _stride_pdfs_0] -
             5.0 * _data_pdfs_2m1_314_10[_stride_pdfs_0 * ctr_0 -
                                         _stride_pdfs_0]);
        const double xi_99 = -_data_pdfs_2m1_311_1m1[_stride_pdfs_0 * ctr_0];
        const double xi_100 = xi_18 + xi_99;
        const double xi_101 =
            -_data_pdfs_20_38_1m1[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_104 =
            -_data_pdfs_2m1_314_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_105 = xi_104 + xi_11 + xi_15 + xi_21;
        const double xi_107 =
            2.0 *
            _data_pdfs_2m1_313_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_108 =
            2.0 *
            _data_pdfs_2m1_314_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_109 =
            2.0 *
                _data_pdfs_21_317_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] +
            2.0 * _data_pdfs_21_318_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_110 =
            omega_even *
            (xi_107 + xi_108 + xi_109 + xi_91 + xi_95 -
             4.0 * _data_pdfs_21_36_10[_stride_pdfs_0 * ctr_0] -
             4.0 * _data_pdfs_2m1_35_10[_stride_pdfs_0 * ctr_0] -
             7.0 *
                 _data_pdfs_20_310_11[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0] -
             7.0 *
                 _data_pdfs_20_37_1m1[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] -
             7.0 *
                 _data_pdfs_20_38_1m1[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0] -
             7.0 *
                 _data_pdfs_20_39_11[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] +
             5.0 * _data_pdfs_20_31_1m1[_stride_pdfs_0 * ctr_0] +
             5.0 * _data_pdfs_20_32_11[_stride_pdfs_0 * ctr_0]);
        const double xi_111 =
            xi_99 + _data_pdfs_2m1_312_11[_stride_pdfs_0 * ctr_0];
        const double xi_112 = xi_111 + xi_14 + xi_22 +
                              _data_pdfs_20_31_1m1[_stride_pdfs_0 * ctr_0] +
                              _data_pdfs_21_316_11[_stride_pdfs_0 * ctr_0];
        const double xi_114 = xi_112 * xi_113;
        const double xi_116 =
            xi_101 +
            _data_pdfs_20_310_11[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_120 = Dummy_298 - 0.5;
        const double xi_125 =
            2.0 * _data_pdfs_20_37_1m1[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_126 =
            2.0 * _data_pdfs_20_310_11[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_127 =
            -2.0 *
                _data_pdfs_20_38_1m1[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0] +
            2.0 * _data_pdfs_20_39_11[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_128 = -xi_125 + xi_126 + xi_127 + xi_14 + xi_19 + xi_2;
        const double xi_130 = xi_128 * xi_129;
        const double xi_131 = Dummy_293 - 0.5;
        const double xi_136 = Dummy_288 - 0.5;
        const double xi_140 =
            _data_pdfs_21_317_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] +
            _data_pdfs_2m1_313_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_154 =
            xi_104 +
            _data_pdfs_2m1_313_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_155 =
            xi_12 + xi_154 + xi_20 +
            _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0] +
            _data_pdfs_21_317_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_156 = xi_113 * xi_155;
        const double xi_157 = Dummy_296 - 0.5;
        const double xi_159 = xi_1 + xi_125 - xi_126 + xi_127 + xi_13;
        const double xi_160 = xi_129 * xi_159;
        const double xi_161 = Dummy_295 - 0.5;
        const double xi_166 = _data_pdfs_21_315_1m1[_stride_pdfs_0 * ctr_0] +
                              _data_pdfs_21_316_11[_stride_pdfs_0 * ctr_0];
        const double xi_167 = xi_100 + xi_166 + xi_23 +
                              _data_pdfs_2m1_35_10[_stride_pdfs_0 * ctr_0];
        const double xi_168 = xi_113 * xi_167;
        const double xi_171 = Dummy_297 - 0.5;
        const double xi_173 = -xi_107 - xi_108 + xi_109 + xi_24 + xi_3;
        const double xi_174 = xi_129 * xi_173;
        const double xi_175 = Dummy_294 - 0.5;
        const double xi_182 = xi_110 * 0.0138888888888889;
        const double xi_203 = xi_96 * -0.00714285714285714;
        const double xi_205 = xi_93 * 0.025;
        const double xi_210 = xi_173 * xi_209;
        const double xi_212 = xi_167 * xi_211;
        const double xi_221 = xi_128 * xi_209;
        const double xi_222 = xi_112 * xi_211;
        const double xi_230 = xi_96 * 0.0178571428571429;
        const double xi_236 = xi_155 * xi_211;
        const double xi_237 = xi_159 * xi_209;
        const double vel0Term =
            xi_1 +
            _data_pdfs_20_310_11[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0] +
            _data_pdfs_20_38_1m1[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double vel1Term =
            xi_2 +
            _data_pdfs_20_37_1m1[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double vel2Term =
            xi_3 +
            _data_pdfs_2m1_313_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double rho = vel0Term + vel1Term + vel2Term + xi_4 + xi_5 + xi_6 +
                           _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0];
        const double xi_7 = 1 / (rho);
        const double xi_84 = rho * temperature;
        const double xi_85 =
            sqrt(xi_84 * (-((-omega_even + 1.0) * (-omega_even + 1.0)) + 1.0));
        const double xi_86 = xi_85 * (Dummy_299 - 0.5) * 3.7416573867739413;
        const double xi_87 = xi_85 * (Dummy_301 - 0.5) * 5.4772255750516612;
        const double xi_89 =
            xi_88 *
            sqrt(xi_84 * (-((-omega_bulk + 1.0) * (-omega_bulk + 1.0)) + 1.0)) *
            (Dummy_292 - 0.5);
        const double xi_90 = xi_85 * (Dummy_300 - 0.5) * 8.3666002653407556;
        const double xi_121 =
            sqrt(xi_84 * (-((-omega_odd + 1.0) * (-omega_odd + 1.0)) + 1.0));
        const double xi_122 = xi_121 * 1.4142135623730951;
        const double xi_123 = xi_122 * 0.5;
        const double xi_124 = xi_120 * xi_123;
        const double xi_132 = xi_121 * xi_88;
        const double xi_133 = xi_132 * 0.166666666666667;
        const double xi_134 = xi_131 * xi_133;
        const double xi_135 = -xi_130 - xi_134;
        const double xi_137 = sqrt(
            xi_84 * (-((-omega_shear + 1.0) * (-omega_shear + 1.0)) + 1.0));
        const double xi_138 = xi_137 * 0.5;
        const double xi_139 = xi_136 * xi_138;
        const double xi_144 =
            xi_110 * -0.0198412698412698 + xi_86 * -0.119047619047619;
        const double xi_146 = xi_137 * (Dummy_287 - 0.5) * 1.7320508075688772;
        const double xi_150 = xi_130 + xi_134;
        const double xi_158 = xi_123 * xi_157;
        const double xi_162 = xi_133 * xi_161;
        const double xi_163 = xi_160 + xi_162;
        const double xi_165 = -xi_160 - xi_162;
        const double xi_172 = xi_123 * xi_171;
        const double xi_176 = xi_133 * xi_175;
        const double xi_177 = -xi_174 - xi_176;
        const double xi_179 = xi_174 + xi_176;
        const double xi_180 = xi_136 * xi_137 * 0.25;
        const double xi_183 = xi_86 * 0.0833333333333333;
        const double xi_193 = xi_138 * (Dummy_289 - 0.5);
        const double xi_202 = xi_138 * (Dummy_291 - 0.5);
        const double xi_206 = xi_90 * -0.0142857142857143;
        const double xi_207 = xi_87 * 0.05;
        const double xi_213 = xi_132 * 0.0833333333333333;
        const double xi_214 = xi_175 * xi_213;
        const double xi_215 = xi_122 * 0.25;
        const double xi_216 = xi_171 * xi_215;
        const double xi_218 =
            xi_110 * -0.00396825396825397 + xi_86 * -0.0238095238095238;
        const double xi_223 = xi_131 * xi_213;
        const double xi_224 = xi_120 * xi_215;
        const double xi_228 = -xi_180;
        const double xi_231 = xi_90 * 0.0357142857142857;
        const double xi_233 = xi_138 * (Dummy_290 - 0.5);
        const double xi_238 = xi_157 * xi_215;
        const double xi_239 = xi_161 * xi_213;
        const double u_0 = xi_7 * (vel0Term + xi_13 + xi_9);
        const double xi_25 =
            u_0 * _data_force_20_30_10[_stride_force_0 * ctr_0];
        const double xi_26 = xi_25 * 0.333333333333333;
        const double xi_32 = -xi_26;
        const double xi_97 = rho * (u_0 * u_0);
        const double xi_151 = rho * u_0;
        const double xi_152 =
            -vel0Term + xi_140 + xi_151 + xi_4 +
            _data_pdfs_20_37_1m1[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        const double xi_153 = xi_118 * xi_152;
        const double xi_189 = xi_152 * xi_184;
        const double u_1 =
            xi_7 *
            (vel1Term + xi_16 + xi_19 + xi_8 +
             _data_pdfs_20_38_1m1[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0]);
        const double xi_27 =
            u_1 * _data_force_20_31_10[_stride_force_0 * ctr_0];
        const double xi_28 = xi_27 * 0.333333333333333;
        const double xi_33 = -xi_28;
        const double xi_54 = u_1 * 0.5;
        const double xi_57 =
            xi_56 * (u_0 * xi_55 +
                     xi_54 * _data_force_20_30_10[_stride_force_0 * ctr_0]);
        const double xi_58 = -xi_57;
        const double xi_102 = rho * (u_1 * u_1);
        const double xi_103 = xi_101 + xi_102 + xi_9;
        const double xi_115 = rho * u_1;
        const double xi_117 =
            -vel1Term + xi_115 + xi_116 + xi_5 +
            _data_pdfs_20_39_11[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] +
            _data_pdfs_2m1_312_11[_stride_pdfs_0 * ctr_0];
        const double xi_119 = xi_117 * xi_118;
        const double xi_185 = xi_117 * xi_184;
        const double xi_195 =
            xi_194 *
            (u_0 * xi_115 + xi_116 + xi_8 +
             _data_pdfs_20_37_1m1[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0]);
        const double xi_196 = -xi_193 - xi_195;
        const double xi_197 = xi_193 + xi_195;
        const double u_2 =
            xi_7 *
            (vel2Term + xi_21 + xi_24 +
             _data_pdfs_2m1_314_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0]);
        const double xi_29 =
            u_2 * _data_force_20_32_10[_stride_force_0 * ctr_0];
        const double xi_30 = xi_29 * 0.333333333333333;
        const double xi_31 = (omega_bulk + 2.0) * (xi_26 + xi_28 + xi_30);
        const double xi_34 = xi_29 * 0.666666666666667 + xi_32 + xi_33;
        const double xi_37 = -xi_30;
        const double xi_38 = xi_27 * 0.666666666666667 + xi_32 + xi_37;
        const double xi_39 = xi_25 * 0.666666666666667 + xi_33 + xi_37;
        const double xi_42 = xi_34 * xi_41;
        const double xi_43 = -xi_42;
        const double xi_44 = xi_39 * xi_41;
        const double xi_45 = -xi_44;
        const double xi_47 = xi_38 * xi_46 + xi_43 + xi_45;
        const double xi_49 = xi_38 * xi_41;
        const double xi_50 = -xi_49;
        const double xi_51 = xi_39 * xi_46 + xi_43 + xi_50;
        const double xi_53 = xi_34 * xi_46 + xi_45 + xi_50;
        const double xi_60 = xi_44 - xi_59;
        const double xi_62 = -xi_34 * xi_61;
        const double xi_64 = xi_31 * 0.125;
        const double xi_65 = xi_49 + xi_64;
        const double xi_66 = xi_63 + xi_65;
        const double xi_67 = xi_62 + xi_66;
        const double xi_68 = xi_44 + xi_59;
        const double xi_69 = -xi_63 + xi_65;
        const double xi_70 = xi_62 + xi_69;
        const double xi_71 =
            xi_56 * (u_2 * xi_55 +
                     xi_54 * _data_force_20_32_10[_stride_force_0 * ctr_0]);
        const double xi_72 = -xi_39 * xi_61;
        const double xi_74 = xi_42 + xi_73;
        const double xi_75 = xi_72 + xi_74;
        const double xi_76 = -xi_71;
        const double xi_77 =
            xi_56 * (u_0 * 0.5 * _data_force_20_32_10[_stride_force_0 * ctr_0] +
                     u_2 * 0.5 * _data_force_20_30_10[_stride_force_0 * ctr_0]);
        const double xi_78 = -xi_77;
        const double xi_79 = -xi_38 * xi_61;
        const double xi_80 = xi_64 + xi_74 + xi_79;
        const double xi_81 = xi_42 - xi_73;
        const double xi_82 = xi_72 + xi_81;
        const double xi_83 = xi_64 + xi_79 + xi_81;
        const double xi_98 = rho * (u_2 * u_2);
        const double xi_106 =
            omega_bulk * (xi_100 + xi_103 + xi_105 + xi_17 + xi_22 + xi_97 +
                          xi_98 + _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0]);
        const double xi_141 = -xi_98 +
                              _data_pdfs_21_36_10[_stride_pdfs_0 * ctr_0] +
                              _data_pdfs_2m1_35_10[_stride_pdfs_0 * ctr_0];
        const double xi_142 =
            omega_shear * (xi_0 + xi_103 + xi_140 + xi_141 + xi_16 -
                           _data_pdfs_20_31_1m1[_stride_pdfs_0 * ctr_0]);
        const double xi_143 = xi_142 * 0.125;
        const double xi_145 =
            omega_shear *
            (xi_101 - xi_102 + xi_105 + xi_141 + xi_9 + xi_94 + xi_97 * 2.0 -
             2.0 *
                 _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0] -
             2.0 *
                 _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0] +
             _data_pdfs_20_31_1m1[_stride_pdfs_0 * ctr_0] +
             _data_pdfs_20_32_11[_stride_pdfs_0 * ctr_0]);
        const double xi_147 =
            xi_145 * -0.0416666666666667 + xi_146 * -0.166666666666667;
        const double xi_148 = xi_147 + xi_87 * -0.1 + xi_93 * -0.05;
        const double xi_149 = xi_139 + xi_143 + xi_144 + xi_148 +
                              xi_90 * 0.0285714285714286 +
                              xi_96 * 0.0142857142857143;
        const double xi_164 =
            xi_144 + xi_145 * 0.0833333333333333 + xi_146 * 0.333333333333333 +
            xi_90 * -0.0714285714285714 + xi_96 * -0.0357142857142857;
        const double xi_169 =
            rho * u_2 - vel2Term + xi_104 + xi_166 + xi_6 + xi_99 +
            _data_pdfs_21_318_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        const double xi_170 = xi_118 * xi_169;
        const double xi_178 = xi_110 * 0.0158730158730159 - xi_139 - xi_143 +
                              xi_148 + xi_86 * 0.0952380952380952 +
                              xi_90 * -0.0428571428571429 +
                              xi_96 * -0.0214285714285714;
        const double xi_181 = xi_142 * 0.0625;
        const double xi_186 =
            xi_106 * 0.0416666666666667 + xi_89 * 0.0833333333333333;
        const double xi_187 = xi_185 + xi_186;
        const double xi_188 =
            xi_150 + xi_180 + xi_181 + xi_182 + xi_183 + xi_187;
        const double xi_190 =
            xi_145 * 0.0208333333333333 + xi_146 * 0.0833333333333333;
        const double xi_191 = -xi_189 + xi_190;
        const double xi_192 = xi_165 + xi_191;
        const double xi_198 = xi_189 + xi_190;
        const double xi_199 = xi_163 + xi_198;
        const double xi_200 = -xi_185 + xi_186;
        const double xi_201 =
            xi_135 + xi_180 + xi_181 + xi_182 + xi_183 + xi_200;
        const double xi_204 =
            xi_194 * (u_2 * xi_115 + xi_111 + xi_17 +
                      _data_pdfs_21_315_1m1[_stride_pdfs_0 * ctr_0]);
        const double xi_208 =
            xi_147 + xi_202 + xi_203 + xi_204 + xi_205 + xi_206 + xi_207;
        const double xi_217 = xi_169 * xi_184;
        const double xi_219 = xi_217 + xi_218;
        const double xi_220 = -xi_210 + xi_212 - xi_214 + xi_216 + xi_219;
        const double xi_225 = xi_187 - xi_221 + xi_222 - xi_223 + xi_224;
        const double xi_226 = xi_200 + xi_221 - xi_222 + xi_223 - xi_224;
        const double xi_227 =
            xi_147 - xi_202 + xi_203 - xi_204 + xi_205 + xi_206 + xi_207;
        const double xi_229 = -xi_181;
        const double xi_232 =
            xi_179 + xi_186 + xi_219 + xi_228 + xi_229 + xi_230 + xi_231;
        const double xi_234 =
            xi_194 *
            (u_2 * xi_151 + xi_10 + xi_154 +
             _data_pdfs_21_318_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0]);
        const double xi_235 = -xi_233 - xi_234;
        const double xi_240 = xi_191 - xi_236 + xi_237 - xi_238 + xi_239;
        const double xi_241 = xi_233 + xi_234;
        const double xi_242 = xi_198 + xi_236 - xi_237 + xi_238 - xi_239;
        const double xi_243 = -xi_217 + xi_218;
        const double xi_244 = xi_210 - xi_212 + xi_214 - xi_216 + xi_243;
        const double xi_245 =
            xi_177 + xi_186 + xi_228 + xi_229 + xi_230 + xi_231 + xi_243;
        const double forceTerm_0 =
            xi_31 * -1.5 - xi_34 * xi_36 - xi_36 * xi_38 - xi_36 * xi_39;
        const double forceTerm_1 = xi_40 + xi_47;
        const double forceTerm_2 = -xi_40 + xi_47;
        const double forceTerm_3 = -xi_48 + xi_51;
        const double forceTerm_4 = xi_48 + xi_51;
        const double forceTerm_5 = xi_52 + xi_53;
        const double forceTerm_6 = -xi_52 + xi_53;
        const double forceTerm_7 = xi_58 + xi_60 + xi_67;
        const double forceTerm_8 = xi_57 + xi_67 + xi_68;
        const double forceTerm_9 = xi_57 + xi_60 + xi_70;
        const double forceTerm_10 = xi_58 + xi_68 + xi_70;
        const double forceTerm_11 = xi_66 + xi_71 + xi_75;
        const double forceTerm_12 = xi_69 + xi_75 + xi_76;
        const double forceTerm_13 = xi_60 + xi_78 + xi_80;
        const double forceTerm_14 = xi_68 + xi_77 + xi_80;
        const double forceTerm_15 = xi_66 + xi_76 + xi_82;
        const double forceTerm_16 = xi_69 + xi_71 + xi_82;
        const double forceTerm_17 = xi_60 + xi_77 + xi_83;
        const double forceTerm_18 = xi_68 + xi_78 + xi_83;
        _data_pdfs_tmp_20_30_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_0 + xi_106 * -0.5 + xi_110 * 0.0238095238095238 +
            xi_86 * 0.142857142857143 + xi_87 * 0.2 - xi_89 +
            xi_90 * 0.0857142857142857 + xi_93 * 0.1 +
            xi_96 * 0.0428571428571429 +
            _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_31_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_1 - xi_114 + xi_119 - xi_124 + xi_135 + xi_149 +
            _data_pdfs_20_31_1m1[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_32_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_2 + xi_114 - xi_119 + xi_124 + xi_149 + xi_150 +
            _data_pdfs_20_32_11[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_33_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_3 - xi_153 + xi_156 + xi_158 + xi_163 + xi_164 +
            _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_34_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_4 + xi_153 - xi_156 - xi_158 + xi_164 + xi_165 +
            _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        _data_pdfs_tmp_20_35_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_5 - xi_168 + xi_170 - xi_172 + xi_177 + xi_178 +
            _data_pdfs_2m1_35_10[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_36_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_6 + xi_168 - xi_170 + xi_172 + xi_178 + xi_179 +
            _data_pdfs_21_36_10[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_37_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_7 + xi_188 + xi_192 + xi_196 +
            _data_pdfs_20_37_1m1[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_38_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_8 + xi_188 + xi_197 + xi_199 +
            _data_pdfs_20_38_1m1[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        _data_pdfs_tmp_20_39_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_9 + xi_192 + xi_197 + xi_201 +
            _data_pdfs_20_39_11[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_310_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_10 + xi_196 + xi_199 + xi_201 +
            _data_pdfs_20_310_11[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        _data_pdfs_tmp_20_311_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_11 + xi_208 + xi_220 + xi_225 +
            _data_pdfs_2m1_311_1m1[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_312_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_12 + xi_220 + xi_226 + xi_227 +
            _data_pdfs_2m1_312_11[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_313_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_13 + xi_232 + xi_235 + xi_240 +
            _data_pdfs_2m1_313_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_314_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_14 + xi_232 + xi_241 + xi_242 +
            _data_pdfs_2m1_314_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        _data_pdfs_tmp_20_315_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_15 + xi_225 + xi_227 + xi_244 +
            _data_pdfs_21_315_1m1[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_316_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_16 + xi_208 + xi_226 + xi_244 +
            _data_pdfs_21_316_11[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_317_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_17 + xi_240 + xi_241 + xi_245 +
            _data_pdfs_21_317_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_318_10[_stride_pdfs_tmp_0 * ctr_0] =
            forceTerm_18 + xi_235 + xi_242 + xi_245 +
            _data_pdfs_21_318_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
      }
    }
  }
}
} // namespace internal_kernel_streamCollide
namespace internal_kernel_collide {
static FUNC_PREFIX void
kernel_collide(double *RESTRICT const _data_force, double *RESTRICT _data_pdfs,
               int64_t const _size_force_0, int64_t const _size_force_1,
               int64_t const _size_force_2, int64_t const _stride_force_0,
               int64_t const _stride_force_1, int64_t const _stride_force_2,
               int64_t const _stride_force_3, int64_t const _stride_pdfs_0,
               int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2,
               int64_t const _stride_pdfs_3, uint32_t block_offset_0,
               uint32_t block_offset_1, uint32_t block_offset_2,
               double omega_bulk, double omega_even, double omega_odd,
               double omega_shear, uint32_t seed, double temperature,
               uint32_t time_step) {
  const double xi_35 = omega_shear + 2.0;
  const double xi_36 = xi_35 * 0.5;
  const double xi_41 = xi_35 * 0.0833333333333333;
  const double xi_46 = xi_35 * 0.166666666666667;
  const double xi_56 = xi_35 * 0.25;
  const double xi_61 = xi_35 * 0.0416666666666667;
  const double xi_88 = 2.4494897427831779;
  const double xi_113 = omega_odd * 0.25;
  const double xi_129 = omega_odd * 0.0833333333333333;
  const double xi_194 = omega_shear * 0.25;
  const double xi_209 = omega_odd * 0.0416666666666667;
  const double xi_211 = omega_odd * 0.125;
  const int64_t rr_0 = 0.0;
  const double xi_118 = rr_0 * 0.166666666666667;
  const double xi_184 = rr_0 * 0.0833333333333333;
  for (int ctr_2 = 0; ctr_2 < _size_force_2; ctr_2 += 1) {
    double *RESTRICT _data_force_20_31 =
        _data_force + _stride_force_2 * ctr_2 + _stride_force_3;
    double *RESTRICT _data_pdfs_20_312 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 12 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_314 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 14 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_31 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_316 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 16 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_32 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 2 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_317 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 17 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_34 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 4 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_310 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 10 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_36 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 6 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_311 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 11 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_318 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 18 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_37 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 7 * _stride_pdfs_3;
    double *RESTRICT _data_force_20_32 =
        _data_force + _stride_force_2 * ctr_2 + 2 * _stride_force_3;
    double *RESTRICT _data_pdfs_20_33 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 3 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_30 = _data_pdfs + _stride_pdfs_2 * ctr_2;
    double *RESTRICT _data_force_20_30 = _data_force + _stride_force_2 * ctr_2;
    double *RESTRICT _data_pdfs_20_313 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 13 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_315 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 15 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_38 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 8 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_35 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 5 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_39 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 9 * _stride_pdfs_3;
    for (int ctr_1 = 0; ctr_1 < _size_force_1; ctr_1 += 1) {
      double *RESTRICT _data_force_20_31_10 =
          _stride_force_1 * ctr_1 + _data_force_20_31;
      double *RESTRICT _data_pdfs_20_312_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_312;
      double *RESTRICT _data_pdfs_20_314_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_314;
      double *RESTRICT _data_pdfs_20_31_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_31;
      double *RESTRICT _data_pdfs_20_316_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_316;
      double *RESTRICT _data_pdfs_20_32_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_32;
      double *RESTRICT _data_pdfs_20_317_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_317;
      double *RESTRICT _data_pdfs_20_34_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_34;
      double *RESTRICT _data_pdfs_20_310_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_310;
      double *RESTRICT _data_pdfs_20_36_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_36;
      double *RESTRICT _data_pdfs_20_311_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_311;
      double *RESTRICT _data_pdfs_20_318_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_318;
      double *RESTRICT _data_pdfs_20_37_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_37;
      double *RESTRICT _data_force_20_32_10 =
          _stride_force_1 * ctr_1 + _data_force_20_32;
      double *RESTRICT _data_pdfs_20_33_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_33;
      double *RESTRICT _data_pdfs_20_30_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_30;
      double *RESTRICT _data_force_20_30_10 =
          _stride_force_1 * ctr_1 + _data_force_20_30;
      double *RESTRICT _data_pdfs_20_313_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_313;
      double *RESTRICT _data_pdfs_20_315_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_315;
      double *RESTRICT _data_pdfs_20_38_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_38;
      double *RESTRICT _data_pdfs_20_35_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_35;
      double *RESTRICT _data_pdfs_20_39_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_39;
      for (int ctr_0 = 0; ctr_0 < _size_force_0; ctr_0 += 1) {
        const double xi_246 = _data_force_20_31_10[_stride_force_0 * ctr_0];
        const double xi_247 = _data_pdfs_20_312_10[_stride_pdfs_0 * ctr_0];
        const double xi_248 = _data_pdfs_20_314_10[_stride_pdfs_0 * ctr_0];
        const double xi_249 = _data_pdfs_20_31_10[_stride_pdfs_0 * ctr_0];
        const double xi_250 = _data_pdfs_20_316_10[_stride_pdfs_0 * ctr_0];
        const double xi_251 = _data_pdfs_20_32_10[_stride_pdfs_0 * ctr_0];
        const double xi_252 = _data_pdfs_20_317_10[_stride_pdfs_0 * ctr_0];
        const double xi_253 = _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0];
        const double xi_254 = _data_pdfs_20_310_10[_stride_pdfs_0 * ctr_0];
        const double xi_255 = _data_pdfs_20_36_10[_stride_pdfs_0 * ctr_0];
        const double xi_256 = _data_pdfs_20_311_10[_stride_pdfs_0 * ctr_0];
        const double xi_257 = _data_pdfs_20_318_10[_stride_pdfs_0 * ctr_0];
        const double xi_258 = _data_pdfs_20_37_10[_stride_pdfs_0 * ctr_0];
        const double xi_259 = _data_force_20_32_10[_stride_force_0 * ctr_0];
        const double xi_260 = _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0];
        const double xi_261 = _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0];
        const double xi_262 = _data_force_20_30_10[_stride_force_0 * ctr_0];
        const double xi_263 = _data_pdfs_20_313_10[_stride_pdfs_0 * ctr_0];
        const double xi_264 = _data_pdfs_20_315_10[_stride_pdfs_0 * ctr_0];
        const double xi_265 = _data_pdfs_20_38_10[_stride_pdfs_0 * ctr_0];
        const double xi_266 = _data_pdfs_20_35_10[_stride_pdfs_0 * ctr_0];
        const double xi_267 = _data_pdfs_20_39_10[_stride_pdfs_0 * ctr_0];

        float Dummy_299;
        float Dummy_300;
        float Dummy_301;
        float Dummy_302;
        philox_float4(time_step, block_offset_0 + ctr_0, block_offset_1 + ctr_1,
                      block_offset_2 + ctr_2, 3, seed, Dummy_299, Dummy_300,
                      Dummy_301, Dummy_302);

        float Dummy_295;
        float Dummy_296;
        float Dummy_297;
        float Dummy_298;
        philox_float4(time_step, block_offset_0 + ctr_0, block_offset_1 + ctr_1,
                      block_offset_2 + ctr_2, 2, seed, Dummy_295, Dummy_296,
                      Dummy_297, Dummy_298);

        float Dummy_291;
        float Dummy_292;
        float Dummy_293;
        float Dummy_294;
        philox_float4(time_step, block_offset_0 + ctr_0, block_offset_1 + ctr_1,
                      block_offset_2 + ctr_2, 1, seed, Dummy_291, Dummy_292,
                      Dummy_293, Dummy_294);

        float Dummy_287;
        float Dummy_288;
        float Dummy_289;
        float Dummy_290;
        philox_float4(time_step, block_offset_0 + ctr_0, block_offset_1 + ctr_1,
                      block_offset_2 + ctr_2, 0, seed, Dummy_287, Dummy_288,
                      Dummy_289, Dummy_290);

        const double xi_0 = xi_248 + xi_257;
        const double xi_1 = xi_0 + xi_253;
        const double xi_2 = xi_249 + xi_256 + xi_264;
        const double xi_3 = xi_247 + xi_266;
        const double xi_4 = xi_260 + xi_267;
        const double xi_5 = xi_250 + xi_251;
        const double xi_6 = xi_252 + xi_255;
        const double xi_8 = -xi_267;
        const double xi_9 = -xi_258 + xi_8;
        const double xi_10 = -xi_252;
        const double xi_11 = -xi_263;
        const double xi_12 = -xi_260;
        const double xi_13 = xi_10 + xi_11 + xi_12;
        const double xi_14 = -xi_251;
        const double xi_15 = -xi_254;
        const double xi_16 = xi_14 + xi_15;
        const double xi_17 = -xi_250;
        const double xi_18 = -xi_247;
        const double xi_19 = xi_17 + xi_18;
        const double xi_20 = -xi_257;
        const double xi_21 = xi_10 + xi_20;
        const double xi_22 = -xi_264;
        const double xi_23 = -xi_255;
        const double xi_24 = xi_17 + xi_22 + xi_23 + xi_256;
        const double xi_40 = xi_246 * 0.166666666666667;
        const double xi_48 = xi_262 * 0.166666666666667;
        const double xi_52 = xi_259 * 0.166666666666667;
        const double xi_55 = xi_246 * 0.5;
        const double xi_59 = xi_262 * 0.0833333333333333;
        const double xi_63 = xi_246 * 0.0833333333333333;
        const double xi_73 = xi_259 * 0.0833333333333333;
        const double xi_91 = -xi_261;
        const double xi_92 = xi_255 * 3.0 + xi_266 * 3.0 + xi_91;
        const double xi_93 =
            omega_even * (xi_247 * -3.0 + xi_249 * 3.0 + xi_250 * -3.0 +
                          xi_251 * 3.0 + xi_256 * -3.0 + xi_264 * -3.0 + xi_92);
        const double xi_94 =
            xi_247 * 2.0 + xi_250 * 2.0 + xi_256 * 2.0 + xi_264 * 2.0;
        const double xi_95 = xi_253 * 5.0 + xi_260 * 5.0 + xi_94;
        const double xi_96 =
            omega_even *
            (xi_248 * -5.0 + xi_249 * -2.0 + xi_251 * -2.0 + xi_252 * -5.0 +
             xi_257 * -5.0 + xi_263 * -5.0 + xi_92 + xi_95);
        const double xi_99 = -xi_256;
        const double xi_100 = xi_18 + xi_99;
        const double xi_101 = -xi_265;
        const double xi_104 = -xi_248;
        const double xi_105 = xi_104 + xi_11 + xi_15 + xi_21;
        const double xi_107 = xi_263 * 2.0;
        const double xi_108 = xi_248 * 2.0;
        const double xi_109 = xi_252 * 2.0 + xi_257 * 2.0;
        const double xi_110 =
            omega_even *
            (xi_107 + xi_108 + xi_109 + xi_249 * 5.0 + xi_251 * 5.0 +
             xi_254 * -7.0 + xi_255 * -4.0 + xi_258 * -7.0 + xi_265 * -7.0 +
             xi_266 * -4.0 + xi_267 * -7.0 + xi_91 + xi_95);
        const double xi_111 = xi_247 + xi_99;
        const double xi_112 = xi_111 + xi_14 + xi_22 + xi_249 + xi_250;
        const double xi_114 = xi_112 * xi_113;
        const double xi_116 = xi_101 + xi_254;
        const double xi_120 = Dummy_298 - 0.5;
        const double xi_125 = xi_258 * 2.0;
        const double xi_126 = xi_254 * 2.0;
        const double xi_127 = xi_265 * -2.0 + xi_267 * 2.0;
        const double xi_128 = -xi_125 + xi_126 + xi_127 + xi_14 + xi_19 + xi_2;
        const double xi_130 = xi_128 * xi_129;
        const double xi_131 = Dummy_293 - 0.5;
        const double xi_136 = Dummy_288 - 0.5;
        const double xi_140 = xi_252 + xi_263;
        const double xi_154 = xi_104 + xi_263;
        const double xi_155 = xi_12 + xi_154 + xi_20 + xi_252 + xi_253;
        const double xi_156 = xi_113 * xi_155;
        const double xi_157 = Dummy_296 - 0.5;
        const double xi_159 = xi_1 + xi_125 - xi_126 + xi_127 + xi_13;
        const double xi_160 = xi_129 * xi_159;
        const double xi_161 = Dummy_295 - 0.5;
        const double xi_166 = xi_250 + xi_264;
        const double xi_167 = xi_100 + xi_166 + xi_23 + xi_266;
        const double xi_168 = xi_113 * xi_167;
        const double xi_171 = Dummy_297 - 0.5;
        const double xi_173 = -xi_107 - xi_108 + xi_109 + xi_24 + xi_3;
        const double xi_174 = xi_129 * xi_173;
        const double xi_175 = Dummy_294 - 0.5;
        const double xi_182 = xi_110 * 0.0138888888888889;
        const double xi_203 = xi_96 * -0.00714285714285714;
        const double xi_205 = xi_93 * 0.025;
        const double xi_210 = xi_173 * xi_209;
        const double xi_212 = xi_167 * xi_211;
        const double xi_221 = xi_128 * xi_209;
        const double xi_222 = xi_112 * xi_211;
        const double xi_230 = xi_96 * 0.0178571428571429;
        const double xi_236 = xi_155 * xi_211;
        const double xi_237 = xi_159 * xi_209;
        const double vel0Term = xi_1 + xi_254 + xi_265;
        const double vel1Term = xi_2 + xi_258;
        const double vel2Term = xi_263 + xi_3;
        const double rho =
            vel0Term + vel1Term + vel2Term + xi_261 + xi_4 + xi_5 + xi_6;
        const double xi_7 = 1 / (rho);
        const double xi_84 = rho * temperature;
        const double xi_85 =
            sqrt(xi_84 * (-((-omega_even + 1.0) * (-omega_even + 1.0)) + 1.0));
        const double xi_86 = xi_85 * (Dummy_299 - 0.5) * 3.7416573867739413;
        const double xi_87 = xi_85 * (Dummy_301 - 0.5) * 5.4772255750516612;
        const double xi_89 =
            xi_88 *
            sqrt(xi_84 * (-((-omega_bulk + 1.0) * (-omega_bulk + 1.0)) + 1.0)) *
            (Dummy_292 - 0.5);
        const double xi_90 = xi_85 * (Dummy_300 - 0.5) * 8.3666002653407556;
        const double xi_121 =
            sqrt(xi_84 * (-((-omega_odd + 1.0) * (-omega_odd + 1.0)) + 1.0));
        const double xi_122 = xi_121 * 1.4142135623730951;
        const double xi_123 = xi_122 * 0.5;
        const double xi_124 = xi_120 * xi_123;
        const double xi_132 = xi_121 * xi_88;
        const double xi_133 = xi_132 * 0.166666666666667;
        const double xi_134 = xi_131 * xi_133;
        const double xi_135 = -xi_130 - xi_134;
        const double xi_137 = sqrt(
            xi_84 * (-((-omega_shear + 1.0) * (-omega_shear + 1.0)) + 1.0));
        const double xi_138 = xi_137 * 0.5;
        const double xi_139 = xi_136 * xi_138;
        const double xi_144 =
            xi_110 * -0.0198412698412698 + xi_86 * -0.119047619047619;
        const double xi_146 = xi_137 * (Dummy_287 - 0.5) * 1.7320508075688772;
        const double xi_150 = xi_130 + xi_134;
        const double xi_158 = xi_123 * xi_157;
        const double xi_162 = xi_133 * xi_161;
        const double xi_163 = xi_160 + xi_162;
        const double xi_165 = -xi_160 - xi_162;
        const double xi_172 = xi_123 * xi_171;
        const double xi_176 = xi_133 * xi_175;
        const double xi_177 = -xi_174 - xi_176;
        const double xi_179 = xi_174 + xi_176;
        const double xi_180 = xi_136 * xi_137 * 0.25;
        const double xi_183 = xi_86 * 0.0833333333333333;
        const double xi_193 = xi_138 * (Dummy_289 - 0.5);
        const double xi_202 = xi_138 * (Dummy_291 - 0.5);
        const double xi_206 = xi_90 * -0.0142857142857143;
        const double xi_207 = xi_87 * 0.05;
        const double xi_213 = xi_132 * 0.0833333333333333;
        const double xi_214 = xi_175 * xi_213;
        const double xi_215 = xi_122 * 0.25;
        const double xi_216 = xi_171 * xi_215;
        const double xi_218 =
            xi_110 * -0.00396825396825397 + xi_86 * -0.0238095238095238;
        const double xi_223 = xi_131 * xi_213;
        const double xi_224 = xi_120 * xi_215;
        const double xi_228 = -xi_180;
        const double xi_231 = xi_90 * 0.0357142857142857;
        const double xi_233 = xi_138 * (Dummy_290 - 0.5);
        const double xi_238 = xi_157 * xi_215;
        const double xi_239 = xi_161 * xi_213;
        const double u_0 = xi_7 * (vel0Term + xi_13 + xi_9);
        const double xi_25 = u_0 * xi_262;
        const double xi_26 = xi_25 * 0.333333333333333;
        const double xi_32 = -xi_26;
        const double xi_97 = rho * (u_0 * u_0);
        const double xi_151 = rho * u_0;
        const double xi_152 = -vel0Term + xi_140 + xi_151 + xi_258 + xi_4;
        const double xi_153 = xi_118 * xi_152;
        const double xi_189 = xi_152 * xi_184;
        const double u_1 = xi_7 * (vel1Term + xi_16 + xi_19 + xi_265 + xi_8);
        const double xi_27 = u_1 * xi_246;
        const double xi_28 = xi_27 * 0.333333333333333;
        const double xi_33 = -xi_28;
        const double xi_54 = u_1 * 0.5;
        const double xi_57 = xi_56 * (u_0 * xi_55 + xi_262 * xi_54);
        const double xi_58 = -xi_57;
        const double xi_102 = rho * (u_1 * u_1);
        const double xi_103 = xi_101 + xi_102 + xi_9;
        const double xi_115 = rho * u_1;
        const double xi_117 =
            -vel1Term + xi_115 + xi_116 + xi_247 + xi_267 + xi_5;
        const double xi_119 = xi_117 * xi_118;
        const double xi_185 = xi_117 * xi_184;
        const double xi_195 = xi_194 * (u_0 * xi_115 + xi_116 + xi_258 + xi_8);
        const double xi_196 = -xi_193 - xi_195;
        const double xi_197 = xi_193 + xi_195;
        const double u_2 = xi_7 * (vel2Term + xi_21 + xi_24 + xi_248);
        const double xi_29 = u_2 * xi_259;
        const double xi_30 = xi_29 * 0.333333333333333;
        const double xi_31 = (omega_bulk + 2.0) * (xi_26 + xi_28 + xi_30);
        const double xi_34 = xi_29 * 0.666666666666667 + xi_32 + xi_33;
        const double xi_37 = -xi_30;
        const double xi_38 = xi_27 * 0.666666666666667 + xi_32 + xi_37;
        const double xi_39 = xi_25 * 0.666666666666667 + xi_33 + xi_37;
        const double xi_42 = xi_34 * xi_41;
        const double xi_43 = -xi_42;
        const double xi_44 = xi_39 * xi_41;
        const double xi_45 = -xi_44;
        const double xi_47 = xi_38 * xi_46 + xi_43 + xi_45;
        const double xi_49 = xi_38 * xi_41;
        const double xi_50 = -xi_49;
        const double xi_51 = xi_39 * xi_46 + xi_43 + xi_50;
        const double xi_53 = xi_34 * xi_46 + xi_45 + xi_50;
        const double xi_60 = xi_44 - xi_59;
        const double xi_62 = -xi_34 * xi_61;
        const double xi_64 = xi_31 * 0.125;
        const double xi_65 = xi_49 + xi_64;
        const double xi_66 = xi_63 + xi_65;
        const double xi_67 = xi_62 + xi_66;
        const double xi_68 = xi_44 + xi_59;
        const double xi_69 = -xi_63 + xi_65;
        const double xi_70 = xi_62 + xi_69;
        const double xi_71 = xi_56 * (u_2 * xi_55 + xi_259 * xi_54);
        const double xi_72 = -xi_39 * xi_61;
        const double xi_74 = xi_42 + xi_73;
        const double xi_75 = xi_72 + xi_74;
        const double xi_76 = -xi_71;
        const double xi_77 = xi_56 * (u_0 * xi_259 * 0.5 + u_2 * xi_262 * 0.5);
        const double xi_78 = -xi_77;
        const double xi_79 = -xi_38 * xi_61;
        const double xi_80 = xi_64 + xi_74 + xi_79;
        const double xi_81 = xi_42 - xi_73;
        const double xi_82 = xi_72 + xi_81;
        const double xi_83 = xi_64 + xi_79 + xi_81;
        const double xi_98 = rho * (u_2 * u_2);
        const double xi_106 = omega_bulk * (xi_100 + xi_103 + xi_105 + xi_17 +
                                            xi_22 + xi_261 + xi_97 + xi_98);
        const double xi_141 = xi_255 + xi_266 - xi_98;
        const double xi_142 =
            omega_shear * (xi_0 + xi_103 + xi_140 + xi_141 + xi_16 - xi_249);
        const double xi_143 = xi_142 * 0.125;
        const double xi_145 =
            omega_shear *
            (xi_101 - xi_102 + xi_105 + xi_141 + xi_249 + xi_251 +
             xi_253 * -2.0 + xi_260 * -2.0 + xi_9 + xi_94 + xi_97 * 2.0);
        const double xi_147 =
            xi_145 * -0.0416666666666667 + xi_146 * -0.166666666666667;
        const double xi_148 = xi_147 + xi_87 * -0.1 + xi_93 * -0.05;
        const double xi_149 = xi_139 + xi_143 + xi_144 + xi_148 +
                              xi_90 * 0.0285714285714286 +
                              xi_96 * 0.0142857142857143;
        const double xi_164 =
            xi_144 + xi_145 * 0.0833333333333333 + xi_146 * 0.333333333333333 +
            xi_90 * -0.0714285714285714 + xi_96 * -0.0357142857142857;
        const double xi_169 =
            rho * u_2 - vel2Term + xi_104 + xi_166 + xi_257 + xi_6 + xi_99;
        const double xi_170 = xi_118 * xi_169;
        const double xi_178 = xi_110 * 0.0158730158730159 - xi_139 - xi_143 +
                              xi_148 + xi_86 * 0.0952380952380952 +
                              xi_90 * -0.0428571428571429 +
                              xi_96 * -0.0214285714285714;
        const double xi_181 = xi_142 * 0.0625;
        const double xi_186 =
            xi_106 * 0.0416666666666667 + xi_89 * 0.0833333333333333;
        const double xi_187 = xi_185 + xi_186;
        const double xi_188 =
            xi_150 + xi_180 + xi_181 + xi_182 + xi_183 + xi_187;
        const double xi_190 =
            xi_145 * 0.0208333333333333 + xi_146 * 0.0833333333333333;
        const double xi_191 = -xi_189 + xi_190;
        const double xi_192 = xi_165 + xi_191;
        const double xi_198 = xi_189 + xi_190;
        const double xi_199 = xi_163 + xi_198;
        const double xi_200 = -xi_185 + xi_186;
        const double xi_201 =
            xi_135 + xi_180 + xi_181 + xi_182 + xi_183 + xi_200;
        const double xi_204 = xi_194 * (u_2 * xi_115 + xi_111 + xi_17 + xi_264);
        const double xi_208 =
            xi_147 + xi_202 + xi_203 + xi_204 + xi_205 + xi_206 + xi_207;
        const double xi_217 = xi_169 * xi_184;
        const double xi_219 = xi_217 + xi_218;
        const double xi_220 = -xi_210 + xi_212 - xi_214 + xi_216 + xi_219;
        const double xi_225 = xi_187 - xi_221 + xi_222 - xi_223 + xi_224;
        const double xi_226 = xi_200 + xi_221 - xi_222 + xi_223 - xi_224;
        const double xi_227 =
            xi_147 - xi_202 + xi_203 - xi_204 + xi_205 + xi_206 + xi_207;
        const double xi_229 = -xi_181;
        const double xi_232 =
            xi_179 + xi_186 + xi_219 + xi_228 + xi_229 + xi_230 + xi_231;
        const double xi_234 = xi_194 * (u_2 * xi_151 + xi_10 + xi_154 + xi_257);
        const double xi_235 = -xi_233 - xi_234;
        const double xi_240 = xi_191 - xi_236 + xi_237 - xi_238 + xi_239;
        const double xi_241 = xi_233 + xi_234;
        const double xi_242 = xi_198 + xi_236 - xi_237 + xi_238 - xi_239;
        const double xi_243 = -xi_217 + xi_218;
        const double xi_244 = xi_210 - xi_212 + xi_214 - xi_216 + xi_243;
        const double xi_245 =
            xi_177 + xi_186 + xi_228 + xi_229 + xi_230 + xi_231 + xi_243;
        const double forceTerm_0 =
            xi_31 * -1.5 - xi_34 * xi_36 - xi_36 * xi_38 - xi_36 * xi_39;
        const double forceTerm_1 = xi_40 + xi_47;
        const double forceTerm_2 = -xi_40 + xi_47;
        const double forceTerm_3 = -xi_48 + xi_51;
        const double forceTerm_4 = xi_48 + xi_51;
        const double forceTerm_5 = xi_52 + xi_53;
        const double forceTerm_6 = -xi_52 + xi_53;
        const double forceTerm_7 = xi_58 + xi_60 + xi_67;
        const double forceTerm_8 = xi_57 + xi_67 + xi_68;
        const double forceTerm_9 = xi_57 + xi_60 + xi_70;
        const double forceTerm_10 = xi_58 + xi_68 + xi_70;
        const double forceTerm_11 = xi_66 + xi_71 + xi_75;
        const double forceTerm_12 = xi_69 + xi_75 + xi_76;
        const double forceTerm_13 = xi_60 + xi_78 + xi_80;
        const double forceTerm_14 = xi_68 + xi_77 + xi_80;
        const double forceTerm_15 = xi_66 + xi_76 + xi_82;
        const double forceTerm_16 = xi_69 + xi_71 + xi_82;
        const double forceTerm_17 = xi_60 + xi_77 + xi_83;
        const double forceTerm_18 = xi_68 + xi_78 + xi_83;
        _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_0 + xi_106 * -0.5 + xi_110 * 0.0238095238095238 + xi_261 +
            xi_86 * 0.142857142857143 + xi_87 * 0.2 - xi_89 +
            xi_90 * 0.0857142857142857 + xi_93 * 0.1 +
            xi_96 * 0.0428571428571429;
        _data_pdfs_20_31_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_1 - xi_114 + xi_119 - xi_124 + xi_135 + xi_149 + xi_249;
        _data_pdfs_20_32_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_2 + xi_114 - xi_119 + xi_124 + xi_149 + xi_150 + xi_251;
        _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_3 - xi_153 + xi_156 + xi_158 + xi_163 + xi_164 + xi_260;
        _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_4 + xi_153 - xi_156 - xi_158 + xi_164 + xi_165 + xi_253;
        _data_pdfs_20_35_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_5 - xi_168 + xi_170 - xi_172 + xi_177 + xi_178 + xi_266;
        _data_pdfs_20_36_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_6 + xi_168 - xi_170 + xi_172 + xi_178 + xi_179 + xi_255;
        _data_pdfs_20_37_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_7 + xi_188 + xi_192 + xi_196 + xi_258;
        _data_pdfs_20_38_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_8 + xi_188 + xi_197 + xi_199 + xi_265;
        _data_pdfs_20_39_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_9 + xi_192 + xi_197 + xi_201 + xi_267;
        _data_pdfs_20_310_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_10 + xi_196 + xi_199 + xi_201 + xi_254;
        _data_pdfs_20_311_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_11 + xi_208 + xi_220 + xi_225 + xi_256;
        _data_pdfs_20_312_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_12 + xi_220 + xi_226 + xi_227 + xi_247;
        _data_pdfs_20_313_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_13 + xi_232 + xi_235 + xi_240 + xi_263;
        _data_pdfs_20_314_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_14 + xi_232 + xi_241 + xi_242 + xi_248;
        _data_pdfs_20_315_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_15 + xi_225 + xi_227 + xi_244 + xi_264;
        _data_pdfs_20_316_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_16 + xi_208 + xi_226 + xi_244 + xi_250;
        _data_pdfs_20_317_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_17 + xi_240 + xi_241 + xi_245 + xi_252;
        _data_pdfs_20_318_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_18 + xi_235 + xi_242 + xi_245 + xi_257;
      }
    }
  }
}
} // namespace internal_kernel_collide
namespace internal_kernel_stream {
static FUNC_PREFIX void kernel_stream(
    double *RESTRICT const _data_pdfs, double *RESTRICT _data_pdfs_tmp,
    int64_t const _size_pdfs_0, int64_t const _size_pdfs_1,
    int64_t const _size_pdfs_2, int64_t const _stride_pdfs_0,
    int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2,
    int64_t const _stride_pdfs_3, int64_t const _stride_pdfs_tmp_0,
    int64_t const _stride_pdfs_tmp_1, int64_t const _stride_pdfs_tmp_2,
    int64_t const _stride_pdfs_tmp_3) {
  for (int ctr_2 = 1; ctr_2 < _size_pdfs_2 - 1; ctr_2 += 1) {
    double *RESTRICT _data_pdfs_tmp_20_30 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2;
    double *RESTRICT _data_pdfs_20_30 = _data_pdfs + _stride_pdfs_2 * ctr_2;
    double *RESTRICT _data_pdfs_tmp_20_31 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_20_31 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_32 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 2 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_20_32 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 2 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_33 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 3 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_20_33 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 3 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_34 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 4 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_20_34 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 4 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_35 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 5 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_2m1_35 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                         _stride_pdfs_2 + 5 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_36 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 6 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_21_36 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                        _stride_pdfs_2 + 6 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_37 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 7 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_20_37 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 7 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_38 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 8 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_20_38 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 8 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_39 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 9 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_20_39 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 9 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_310 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 10 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_20_310 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 10 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_311 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 11 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_2m1_311 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                          _stride_pdfs_2 + 11 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_312 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 12 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_2m1_312 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                          _stride_pdfs_2 + 12 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_313 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 13 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_2m1_313 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                          _stride_pdfs_2 + 13 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_314 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 14 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_2m1_314 = _data_pdfs + _stride_pdfs_2 * ctr_2 -
                                          _stride_pdfs_2 + 14 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_315 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 15 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_21_315 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                         _stride_pdfs_2 + 15 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_316 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 16 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_21_316 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                         _stride_pdfs_2 + 16 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_317 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 17 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_21_317 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                         _stride_pdfs_2 + 17 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_tmp_20_318 =
        _data_pdfs_tmp + _stride_pdfs_tmp_2 * ctr_2 + 18 * _stride_pdfs_tmp_3;
    double *RESTRICT _data_pdfs_21_318 = _data_pdfs + _stride_pdfs_2 * ctr_2 +
                                         _stride_pdfs_2 + 18 * _stride_pdfs_3;
    for (int ctr_1 = 1; ctr_1 < _size_pdfs_1 - 1; ctr_1 += 1) {
      double *RESTRICT _data_pdfs_tmp_20_30_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_30;
      double *RESTRICT _data_pdfs_20_30_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_30;
      double *RESTRICT _data_pdfs_tmp_20_31_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_31;
      double *RESTRICT _data_pdfs_20_31_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_20_31;
      double *RESTRICT _data_pdfs_tmp_20_32_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_32;
      double *RESTRICT _data_pdfs_20_32_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_20_32;
      double *RESTRICT _data_pdfs_tmp_20_33_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_33;
      double *RESTRICT _data_pdfs_20_33_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_33;
      double *RESTRICT _data_pdfs_tmp_20_34_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_34;
      double *RESTRICT _data_pdfs_20_34_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_34;
      double *RESTRICT _data_pdfs_tmp_20_35_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_35;
      double *RESTRICT _data_pdfs_2m1_35_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_2m1_35;
      double *RESTRICT _data_pdfs_tmp_20_36_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_36;
      double *RESTRICT _data_pdfs_21_36_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_21_36;
      double *RESTRICT _data_pdfs_tmp_20_37_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_37;
      double *RESTRICT _data_pdfs_20_37_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_20_37;
      double *RESTRICT _data_pdfs_tmp_20_38_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_38;
      double *RESTRICT _data_pdfs_20_38_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_20_38;
      double *RESTRICT _data_pdfs_tmp_20_39_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_39;
      double *RESTRICT _data_pdfs_20_39_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_20_39;
      double *RESTRICT _data_pdfs_tmp_20_310_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_310;
      double *RESTRICT _data_pdfs_20_310_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_20_310;
      double *RESTRICT _data_pdfs_tmp_20_311_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_311;
      double *RESTRICT _data_pdfs_2m1_311_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_2m1_311;
      double *RESTRICT _data_pdfs_tmp_20_312_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_312;
      double *RESTRICT _data_pdfs_2m1_312_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_2m1_312;
      double *RESTRICT _data_pdfs_tmp_20_313_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_313;
      double *RESTRICT _data_pdfs_2m1_313_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_2m1_313;
      double *RESTRICT _data_pdfs_tmp_20_314_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_314;
      double *RESTRICT _data_pdfs_2m1_314_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_2m1_314;
      double *RESTRICT _data_pdfs_tmp_20_315_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_315;
      double *RESTRICT _data_pdfs_21_315_1m1 =
          _stride_pdfs_1 * ctr_1 - _stride_pdfs_1 + _data_pdfs_21_315;
      double *RESTRICT _data_pdfs_tmp_20_316_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_316;
      double *RESTRICT _data_pdfs_21_316_11 =
          _stride_pdfs_1 * ctr_1 + _stride_pdfs_1 + _data_pdfs_21_316;
      double *RESTRICT _data_pdfs_tmp_20_317_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_317;
      double *RESTRICT _data_pdfs_21_317_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_21_317;
      double *RESTRICT _data_pdfs_tmp_20_318_10 =
          _stride_pdfs_tmp_1 * ctr_1 + _data_pdfs_tmp_20_318;
      double *RESTRICT _data_pdfs_21_318_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_21_318;
      for (int ctr_0 = 1; ctr_0 < _size_pdfs_0 - 1; ctr_0 += 1) {
        _data_pdfs_tmp_20_30_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_31_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_20_31_1m1[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_32_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_20_32_11[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_33_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_34_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        _data_pdfs_tmp_20_35_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_2m1_35_10[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_36_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_21_36_10[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_37_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_20_37_1m1[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_38_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_20_38_1m1[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        _data_pdfs_tmp_20_39_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_20_39_11[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_310_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_20_310_11[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        _data_pdfs_tmp_20_311_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_2m1_311_1m1[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_312_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_2m1_312_11[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_313_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_2m1_313_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_314_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_2m1_314_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
        _data_pdfs_tmp_20_315_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_21_315_1m1[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_316_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_21_316_11[_stride_pdfs_0 * ctr_0];
        _data_pdfs_tmp_20_317_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_21_317_10[_stride_pdfs_0 * ctr_0 + _stride_pdfs_0];
        _data_pdfs_tmp_20_318_10[_stride_pdfs_tmp_0 * ctr_0] =
            _data_pdfs_21_318_10[_stride_pdfs_0 * ctr_0 - _stride_pdfs_0];
      }
    }
  }
}
} // namespace internal_kernel_stream

const real_t FluctuatingMRT_LatticeModel::w[19] = {
    0.333333333333333,  0.0555555555555556, 0.0555555555555556,
    0.0555555555555556, 0.0555555555555556, 0.0555555555555556,
    0.0555555555555556, 0.0277777777777778, 0.0277777777777778,
    0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
    0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
    0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
    0.0277777777777778};
const real_t FluctuatingMRT_LatticeModel::wInv[19] = {
    3.00000000000000, 18.0000000000000, 18.0000000000000, 18.0000000000000,
    18.0000000000000, 18.0000000000000, 18.0000000000000, 36.0000000000000,
    36.0000000000000, 36.0000000000000, 36.0000000000000, 36.0000000000000,
    36.0000000000000, 36.0000000000000, 36.0000000000000, 36.0000000000000,
    36.0000000000000, 36.0000000000000, 36.0000000000000};

void FluctuatingMRT_LatticeModel::Sweep::streamCollide(
    IBlock *block, const uint_t numberOfGhostLayersToInclude) {
  auto pdfs = block->getData<field::GhostLayerField<double, 19>>(pdfsID);
  field::GhostLayerField<double, 19> *pdfs_tmp;
  {
    // Getting temporary field pdfs_tmp
    auto it = cache_pdfs_.find(pdfs);
    if (it != cache_pdfs_.end()) {
      pdfs_tmp = *it;
    } else {
      pdfs_tmp = pdfs->cloneUninitialized();
      cache_pdfs_.insert(pdfs_tmp);
    }
  }

  auto &lm = dynamic_cast<lbm::PdfField<FluctuatingMRT_LatticeModel> *>(pdfs)
                 ->latticeModel();
  WALBERLA_ASSERT_EQUAL(*(lm.blockId_), block->getId());

  auto &omega_shear = lm.omega_shear_;
  auto &seed = lm.seed_;
  auto &force = lm.force_;
  auto &block_offset_1 = lm.block_offset_1_;
  auto &omega_odd = lm.omega_odd_;
  auto &omega_bulk = lm.omega_bulk_;
  auto &block_offset_0 = lm.block_offset_0_;
  auto &omega_even = lm.omega_even_;
  auto &temperature = lm.temperature_;
  auto &time_step = lm.time_step_;
  auto &block_offset_2 = lm.block_offset_2_;
  WALBERLA_ASSERT_GREATER_EQUAL(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                                -int_c(force->nrOfGhostLayers()));
  double *RESTRICT const _data_force =
      force->dataAt(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                    -cell_idx_c(numberOfGhostLayersToInclude) - 1,
                    -cell_idx_c(numberOfGhostLayersToInclude) - 1, 0);
  WALBERLA_ASSERT_GREATER_EQUAL(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                                -int_c(pdfs->nrOfGhostLayers()));
  double *RESTRICT const _data_pdfs =
      pdfs->dataAt(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                   -cell_idx_c(numberOfGhostLayersToInclude) - 1,
                   -cell_idx_c(numberOfGhostLayersToInclude) - 1, 0);
  WALBERLA_ASSERT_GREATER_EQUAL(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                                -int_c(pdfs_tmp->nrOfGhostLayers()));
  double *RESTRICT _data_pdfs_tmp =
      pdfs_tmp->dataAt(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                       -cell_idx_c(numberOfGhostLayersToInclude) - 1,
                       -cell_idx_c(numberOfGhostLayersToInclude) - 1, 0);
  WALBERLA_ASSERT_GREATER_EQUAL(
      force->xSizeWithGhostLayer(),
      int64_t(cell_idx_c(force->xSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2));
  const int64_t _size_force_0 =
      int64_t(cell_idx_c(force->xSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2);
  WALBERLA_ASSERT_GREATER_EQUAL(
      force->ySizeWithGhostLayer(),
      int64_t(cell_idx_c(force->ySize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2));
  const int64_t _size_force_1 =
      int64_t(cell_idx_c(force->ySize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2);
  WALBERLA_ASSERT_GREATER_EQUAL(
      force->zSizeWithGhostLayer(),
      int64_t(cell_idx_c(force->zSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2));
  const int64_t _size_force_2 =
      int64_t(cell_idx_c(force->zSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2);
  const int64_t _stride_force_0 = int64_t(force->xStride());
  const int64_t _stride_force_1 = int64_t(force->yStride());
  const int64_t _stride_force_2 = int64_t(force->zStride());
  const int64_t _stride_force_3 = int64_t(1 * int64_t(force->fStride()));
  const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
  const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
  const int64_t _stride_pdfs_2 = int64_t(pdfs->zStride());
  const int64_t _stride_pdfs_3 = int64_t(1 * int64_t(pdfs->fStride()));
  const int64_t _stride_pdfs_tmp_0 = int64_t(pdfs_tmp->xStride());
  const int64_t _stride_pdfs_tmp_1 = int64_t(pdfs_tmp->yStride());
  const int64_t _stride_pdfs_tmp_2 = int64_t(pdfs_tmp->zStride());
  const int64_t _stride_pdfs_tmp_3 = int64_t(1 * int64_t(pdfs_tmp->fStride()));
  internal_kernel_streamCollide::kernel_streamCollide(
      _data_force, _data_pdfs, _data_pdfs_tmp, _size_force_0, _size_force_1,
      _size_force_2, _stride_force_0, _stride_force_1, _stride_force_2,
      _stride_force_3, _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2,
      _stride_pdfs_3, _stride_pdfs_tmp_0, _stride_pdfs_tmp_1,
      _stride_pdfs_tmp_2, _stride_pdfs_tmp_3, block_offset_0, block_offset_1,
      block_offset_2, omega_bulk, omega_even, omega_odd, omega_shear, seed,
      temperature, time_step);
  pdfs->swapDataPointers(pdfs_tmp);
}

void FluctuatingMRT_LatticeModel::Sweep::collide(
    IBlock *block, const uint_t numberOfGhostLayersToInclude) {
  auto pdfs = block->getData<field::GhostLayerField<double, 19>>(pdfsID);

  auto &lm = dynamic_cast<lbm::PdfField<FluctuatingMRT_LatticeModel> *>(pdfs)
                 ->latticeModel();
  WALBERLA_ASSERT_EQUAL(*(lm.blockId_), block->getId());

  auto &omega_shear = lm.omega_shear_;
  auto &seed = lm.seed_;
  auto &force = lm.force_;
  auto &block_offset_1 = lm.block_offset_1_;
  auto &omega_odd = lm.omega_odd_;
  auto &omega_bulk = lm.omega_bulk_;
  auto &block_offset_0 = lm.block_offset_0_;
  auto &omega_even = lm.omega_even_;
  auto &temperature = lm.temperature_;
  auto &time_step = lm.time_step_;
  auto &block_offset_2 = lm.block_offset_2_;
  WALBERLA_ASSERT_GREATER_EQUAL(-cell_idx_c(numberOfGhostLayersToInclude),
                                -int_c(force->nrOfGhostLayers()));
  double *RESTRICT const _data_force =
      force->dataAt(-cell_idx_c(numberOfGhostLayersToInclude),
                    -cell_idx_c(numberOfGhostLayersToInclude),
                    -cell_idx_c(numberOfGhostLayersToInclude), 0);
  WALBERLA_ASSERT_GREATER_EQUAL(-cell_idx_c(numberOfGhostLayersToInclude),
                                -int_c(pdfs->nrOfGhostLayers()));
  double *RESTRICT _data_pdfs =
      pdfs->dataAt(-cell_idx_c(numberOfGhostLayersToInclude),
                   -cell_idx_c(numberOfGhostLayersToInclude),
                   -cell_idx_c(numberOfGhostLayersToInclude), 0);
  WALBERLA_ASSERT_GREATER_EQUAL(
      force->xSizeWithGhostLayer(),
      int64_t(cell_idx_c(force->xSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude)));
  const int64_t _size_force_0 =
      int64_t(cell_idx_c(force->xSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude));
  WALBERLA_ASSERT_GREATER_EQUAL(
      force->ySizeWithGhostLayer(),
      int64_t(cell_idx_c(force->ySize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude)));
  const int64_t _size_force_1 =
      int64_t(cell_idx_c(force->ySize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude));
  WALBERLA_ASSERT_GREATER_EQUAL(
      force->zSizeWithGhostLayer(),
      int64_t(cell_idx_c(force->zSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude)));
  const int64_t _size_force_2 =
      int64_t(cell_idx_c(force->zSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude));
  const int64_t _stride_force_0 = int64_t(force->xStride());
  const int64_t _stride_force_1 = int64_t(force->yStride());
  const int64_t _stride_force_2 = int64_t(force->zStride());
  const int64_t _stride_force_3 = int64_t(1 * int64_t(force->fStride()));
  const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
  const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
  const int64_t _stride_pdfs_2 = int64_t(pdfs->zStride());
  const int64_t _stride_pdfs_3 = int64_t(1 * int64_t(pdfs->fStride()));
  internal_kernel_collide::kernel_collide(
      _data_force, _data_pdfs, _size_force_0, _size_force_1, _size_force_2,
      _stride_force_0, _stride_force_1, _stride_force_2, _stride_force_3,
      _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_pdfs_3,
      block_offset_0, block_offset_1, block_offset_2, omega_bulk, omega_even,
      omega_odd, omega_shear, seed, temperature, time_step);
}

void FluctuatingMRT_LatticeModel::Sweep::stream(
    IBlock *block, const uint_t numberOfGhostLayersToInclude) {
  auto pdfs = block->getData<field::GhostLayerField<double, 19>>(pdfsID);
  field::GhostLayerField<double, 19> *pdfs_tmp;
  {
    // Getting temporary field pdfs_tmp
    auto it = cache_pdfs_.find(pdfs);
    if (it != cache_pdfs_.end()) {
      pdfs_tmp = *it;
    } else {
      pdfs_tmp = pdfs->cloneUninitialized();
      cache_pdfs_.insert(pdfs_tmp);
    }
  }

  WALBERLA_ASSERT_GREATER_EQUAL(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                                -int_c(pdfs->nrOfGhostLayers()));
  double *RESTRICT const _data_pdfs =
      pdfs->dataAt(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                   -cell_idx_c(numberOfGhostLayersToInclude) - 1,
                   -cell_idx_c(numberOfGhostLayersToInclude) - 1, 0);
  WALBERLA_ASSERT_GREATER_EQUAL(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                                -int_c(pdfs_tmp->nrOfGhostLayers()));
  double *RESTRICT _data_pdfs_tmp =
      pdfs_tmp->dataAt(-cell_idx_c(numberOfGhostLayersToInclude) - 1,
                       -cell_idx_c(numberOfGhostLayersToInclude) - 1,
                       -cell_idx_c(numberOfGhostLayersToInclude) - 1, 0);
  WALBERLA_ASSERT_GREATER_EQUAL(
      pdfs->xSizeWithGhostLayer(),
      int64_t(cell_idx_c(pdfs->xSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2));
  const int64_t _size_pdfs_0 =
      int64_t(cell_idx_c(pdfs->xSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2);
  WALBERLA_ASSERT_GREATER_EQUAL(
      pdfs->ySizeWithGhostLayer(),
      int64_t(cell_idx_c(pdfs->ySize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2));
  const int64_t _size_pdfs_1 =
      int64_t(cell_idx_c(pdfs->ySize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2);
  WALBERLA_ASSERT_GREATER_EQUAL(
      pdfs->zSizeWithGhostLayer(),
      int64_t(cell_idx_c(pdfs->zSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2));
  const int64_t _size_pdfs_2 =
      int64_t(cell_idx_c(pdfs->zSize()) +
              2 * cell_idx_c(numberOfGhostLayersToInclude) + 2);
  const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
  const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
  const int64_t _stride_pdfs_2 = int64_t(pdfs->zStride());
  const int64_t _stride_pdfs_3 = int64_t(1 * int64_t(pdfs->fStride()));
  const int64_t _stride_pdfs_tmp_0 = int64_t(pdfs_tmp->xStride());
  const int64_t _stride_pdfs_tmp_1 = int64_t(pdfs_tmp->yStride());
  const int64_t _stride_pdfs_tmp_2 = int64_t(pdfs_tmp->zStride());
  const int64_t _stride_pdfs_tmp_3 = int64_t(1 * int64_t(pdfs_tmp->fStride()));
  internal_kernel_stream::kernel_stream(
      _data_pdfs, _data_pdfs_tmp, _size_pdfs_0, _size_pdfs_1, _size_pdfs_2,
      _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_pdfs_3,
      _stride_pdfs_tmp_0, _stride_pdfs_tmp_1, _stride_pdfs_tmp_2,
      _stride_pdfs_tmp_3);

  pdfs->swapDataPointers(pdfs_tmp);
}

} // namespace lbm
} // namespace walberla

// Buffer Packing

namespace walberla {
namespace mpi {

mpi::SendBuffer &
operator<<(mpi::SendBuffer &buf,
           const ::walberla::lbm::FluctuatingMRT_LatticeModel &lm) {
  buf << lm.currentLevel;
  return buf;
}

mpi::RecvBuffer &operator>>(mpi::RecvBuffer &buf,
                            ::walberla::lbm::FluctuatingMRT_LatticeModel &lm) {
  buf >> lm.currentLevel;
  return buf;
}

} // namespace mpi
} // namespace walberla

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic pop
#endif