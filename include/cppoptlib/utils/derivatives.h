// CPPNumericalSolvers - A lightweight C++ numerical optimization library
// Copyright (c) 2014    Patrick Wieschollek + Contributors
// Licensed under the MIT License (see below).
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Author: Patrick Wieschollek
//
// This file contains implementation details of numerical optimization solvers
// using automatic differentiation and gradient-based methods.
// CPPNumericalSolvers provides a flexible and efficient interface for solving
// unconstrained and constrained optimization problems.
//
// More details can be found in the project documentation:
// https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_
#define INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace cppoptlib::utils {

template <class FunctionType>
void ComputeFiniteGradient(
    const FunctionType &function,
    const Eigen::Matrix<typename FunctionType::ScalarType, Eigen::Dynamic, 1>
        &x0,
    Eigen::Matrix<typename FunctionType::ScalarType, Eigen::Dynamic, 1> *grad,
    const int accuracy = 0) {
  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using index_t = typename VectorType::Index;

  constexpr ScalarType machine_eps = std::numeric_limits<ScalarType>::epsilon();

  // Coefficients for finite difference formulas of increasing accuracy.
  // The 'accuracy' can be 0, 1, 2, or 3.
  static const std::array<std::vector<ScalarType>, 4> coeff = {
      {{1, -1},
       {1, -8, 8, -1},
       {-1, 9, -45, 45, -9, 1},
       {3, -32, 168, -672, 672, -168, 32, -3}}};
  static const std::array<std::vector<ScalarType>, 4> coeff2 = {
      {{1, -1},
       {-2, -1, 1, 2},
       {-3, -2, -1, 1, 2, 3},
       {-4, -3, -2, -1, 1, 2, 3, 4}}};

  static const std::array<ScalarType, 4> dd = {2, 12, 60, 840};

  grad->resize(x0.rows());
  VectorType x = x0;

  const int innerSteps = 2 * (accuracy + 1);
  for (index_t d = 0; d < x0.rows(); d++) {
    // Compute a coordinate-dependent step size.
    ScalarType h =
        std::sqrt(machine_eps) * std::max(std::abs(x0[d]), ScalarType(1));
    ScalarType ddVal = dd[accuracy] * h;
    (*grad)[d] = 0;
    for (int s = 0; s < innerSteps; ++s) {
      ScalarType tmp = x[d];
      x[d] += coeff2[accuracy][s] * h;
      (*grad)[d] += coeff[accuracy][s] * function(x);
      x[d] = tmp;
    }
    (*grad)[d] /= ddVal;
  }
}

// Approximates the Hessian of the given function in x0.
template <class FunctionType>
void ComputeFiniteHessian(
    const FunctionType &function,
    const Eigen::Matrix<typename FunctionType::ScalarType, Eigen::Dynamic, 1>
        &x0,
    Eigen::Matrix<typename FunctionType::ScalarType, Eigen::Dynamic,
                  Eigen::Dynamic> *hessian,
    int accuracy = 0) {
  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;
  using index_t = typename VectorType::Index;

  constexpr ScalarType machine_eps = std::numeric_limits<ScalarType>::epsilon();
  index_t n = x0.rows();
  hessian->resize(n, n);

  // Make a local copy so we can safely modify it.
  VectorType x = x0;
  ScalarType f0 = function(x0);

  if (accuracy == 0) {
    // Basic central difference approximation.
    for (index_t i = 0; i < n; i++) {
      // Choose an adaptive step size for the i-th coordinate.
      ScalarType hi =
          std::sqrt(machine_eps) * std::max(std::abs(x0[i]), ScalarType(1));
      // Diagonal: standard second derivative.
      x = x0;
      x[i] += hi;
      ScalarType f_plus = function(x);
      x = x0;
      x[i] -= hi;
      ScalarType f_minus = function(x);
      (*hessian)(i, i) = (f_plus - 2 * f0 + f_minus) / (hi * hi);

      for (index_t j = i + 1; j < n; j++) {
        ScalarType hj =
            std::sqrt(machine_eps) * std::max(std::abs(x0[j]), ScalarType(1));
        // Off-diagonals: use central differences.
        x = x0;
        x[i] += hi;
        x[j] += hj;
        ScalarType f_pp = function(x);
        x = x0;
        x[i] += hi;
        x[j] -= hj;
        ScalarType f_pm = function(x);
        x = x0;
        x[i] -= hi;
        x[j] += hj;
        ScalarType f_mp = function(x);
        x = x0;
        x[i] -= hi;
        x[j] -= hj;
        ScalarType f_mm = function(x);
        ScalarType d2f = (f_pp - f_pm - f_mp + f_mm) / (4 * hi * hj);
        (*hessian)(i, j) = d2f;
        (*hessian)(j, i) = d2f;
      }
    }
  } else {
    // Higher-order finite difference approximation using
    for (index_t i = 0; i < n; i++) {
      ScalarType hi =
          std::sqrt(machine_eps) * std::max(std::abs(x0[i]), ScalarType(1));
      x = x0;
      x[i] += hi;
      ScalarType f_plus = function(x);
      x = x0;
      x[i] -= hi;
      ScalarType f_minus = function(x);
      (*hessian)(i, i) = (f_plus - 2 * f0 + f_minus) / (hi * hi);

      for (index_t j = i + 1; j < n; j++) {
        ScalarType hj =
            std::sqrt(machine_eps) * std::max(std::abs(x0[j]), ScalarType(1));
        ScalarType h = (hi + hj) / 2;

        ScalarType tmpi = x0[i], tmpj = x0[j];
        ScalarType term1 = 0, term2 = 0, term3 = 0, term4 = 0;

        // term1: -63 * (f(tmpi+1h, tmpj-2h) + f(tmpi+2h, tmpj-1h) + f(tmpi-2h,
        // tmpj+1h) + f(tmpi-1h, tmpj+2h))
        x = x0;
        x[i] = tmpi + 1 * h;
        x[j] = tmpj - 2 * h;
        term1 += function(x);
        x = x0;
        x[i] = tmpi + 2 * h;
        x[j] = tmpj - 1 * h;
        term1 += function(x);
        x = x0;
        x[i] = tmpi - 2 * h;
        x[j] = tmpj + 1 * h;
        term1 += function(x);
        x = x0;
        x[i] = tmpi - 1 * h;
        x[j] = tmpj + 2 * h;
        term1 += function(x);

        // term2: +63 * (f(tmpi-1h, tmpj-2h) + f(tmpi-2h, tmpj-1h) + f(tmpi+1h,
        // tmpj+2h) + f(tmpi+2h, tmpj+1h))
        x = x0;
        x[i] = tmpi - 1 * h;
        x[j] = tmpj - 2 * h;
        term2 += function(x);
        x = x0;
        x[i] = tmpi - 2 * h;
        x[j] = tmpj - 1 * h;
        term2 += function(x);
        x = x0;
        x[i] = tmpi + 1 * h;
        x[j] = tmpj + 2 * h;
        term2 += function(x);
        x = x0;
        x[i] = tmpi + 2 * h;
        x[j] = tmpj + 1 * h;
        term2 += function(x);

        // term3: +44 * (f(tmpi+2h, tmpj-2h) + f(tmpi-2h, tmpj+2h) - f(tmpi-2h,
        // tmpj-2h) - f(tmpi+2h, tmpj+2h))
        x = x0;
        x[i] = tmpi + 2 * h;
        x[j] = tmpj - 2 * h;
        term3 += function(x);
        x = x0;
        x[i] = tmpi - 2 * h;
        x[j] = tmpj + 2 * h;
        term3 += function(x);
        x = x0;
        x[i] = tmpi - 2 * h;
        x[j] = tmpj - 2 * h;
        term3 -= function(x);
        x = x0;
        x[i] = tmpi + 2 * h;
        x[j] = tmpj + 2 * h;
        term3 -= function(x);

        // term4: +74 * (f(tmpi-1h, tmpj-1h) + f(tmpi+1h, tmpj+1h) - f(tmpi+1h,
        // tmpj-1h) - f(tmpi-1h, tmpj+1h))
        x = x0;
        x[i] = tmpi - 1 * h;
        x[j] = tmpj - 1 * h;
        term4 += function(x);
        x = x0;
        x[i] = tmpi + 1 * h;
        x[j] = tmpj + 1 * h;
        term4 += function(x);
        x = x0;
        x[i] = tmpi + 1 * h;
        x[j] = tmpj - 1 * h;
        term4 -= function(x);
        x = x0;
        x[i] = tmpi - 1 * h;
        x[j] = tmpj + 1 * h;
        term4 -= function(x);

        // Combine the weighted terms.
        ScalarType mixed =
            (-63 * term1 + 63 * term2 + 44 * term3 + 74 * term4) /
            (600 * h * h);
        (*hessian)(i, j) = mixed;
        (*hessian)(j, i) = mixed;
      }
    }
  }
}

template <class FunctionType>
bool IsGradientCorrect(const FunctionType &function,
                       const typename FunctionType::VectorType &x0,
                       int accuracy = 3) {
  constexpr float tolerance = 1e-2;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using index_t = typename VectorType::Index;

  const index_t D = x0.rows();
  VectorType actual_gradient;
  function(x0, &actual_gradient);
  VectorType expected_gradient(D);

  ComputeFiniteGradient(function, x0, &expected_gradient, accuracy);

  for (index_t d = 0; d < D; ++d) {
    ScalarType scale =
        std::max(static_cast<ScalarType>(std::max(fabs(actual_gradient[d]),
                                                  fabs(expected_gradient[d]))),
                 ScalarType(1.));
    if (fabs(actual_gradient[d] - expected_gradient[d]) > tolerance * scale)
      return false;
  }
  return true;
}

template <class FunctionType>
bool IsHessianCorrect(const FunctionType &function,
                      const typename FunctionType::VectorType &x0,
                      int accuracy = 3) {
  constexpr float tolerance = 1e-1;

  using ScalarType = typename FunctionType::ScalarType;
  using MatrixType = typename FunctionType::MatrixType;
  using VectorType = typename FunctionType::VectorType;
  using index_t = typename VectorType::Index;

  const index_t D = x0.rows();

  MatrixType actual_hessian;
  function(x0, nullptr, &actual_hessian);
  MatrixType expected_hessian = MatrixType::Zero(D, D);
  ComputeFiniteHessian(function, x0, &expected_hessian, accuracy);
  for (index_t d = 0; d < D; ++d) {
    for (index_t e = 0; e < D; ++e) {
      ScalarType scale = std::max(
          static_cast<ScalarType>(std::max(fabs(actual_hessian(d, e)),
                                           fabs(expected_hessian(d, e)))),
          ScalarType(1.));
      if (fabs(actual_hessian(d, e) - expected_hessian(d, e)) >
          tolerance * scale)
        return false;
    }
  }
  return true;
}

}  // namespace cppoptlib::utils

#endif  // INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_
