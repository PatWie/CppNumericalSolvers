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
// More details can be found in the project documentation:
// https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_NELDER_MEAD_H_
#define INCLUDE_CPPOPTLIB_SOLVER_NELDER_MEAD_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib::solver {

template <typename FunctionType>
class NelderMead
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using progress_t = typename Superclass::progress_t;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using MatrixType = typename FunctionType::MatrixType;

  // Coefficients for the Nelder–Mead algorithm.
  const ScalarType rho_ = 1.0;    // Reflection coefficient (> 0)
  const ScalarType xi_ = 20.0;    // Expansion coefficient
  const ScalarType gamma_ = 0.1;  // Contraction coefficient (0 < gamma < 1)
  const ScalarType sigma_ = 0.5;  // Shrink coefficient (0 < sigma < 1)

  // Tolerance for detecting a degenerate simplex.
  const ScalarType degenerate_tol_ = 1e-8;

  using Superclass::Superclass;

  // Initialize the solver with the starting point.
  void InitializeSolver(const FunctionType & /*function*/,
                        const StateType &initial_state) override {
    simplex_ = makeInitialSimplex(initial_state.x);
  }

  // Performs one iteration (step) of the Nelder–Mead algorithm.
  // This implementation updates the internal simplex and returns the current
  // best vertex.
  StateType OptimizationStep(const FunctionType &function,
                             const StateType &current,
                             const progress_t & /*progress*/) override {
    const size_t DIM = current.x.rows();
    const int numVertices = static_cast<int>(DIM) + 1;

    // Evaluate the objective function at each vertex of the simplex.
    std::vector<ScalarType> f(numVertices);
    std::vector<int> idx(numVertices);
    for (int i = 0; i < numVertices; ++i) {
      f[i] = function(simplex_.col(i));
      idx[i] = i;
    }

    // Sort indices so that f[idx[0]] is the best (lowest) function value.
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return f[a] < f[b]; });

    // Check for degeneracy: if the maximum distance between the best and other
    // vertices is too small, restart the simplex around the best vertex.
    ScalarType maxDist = 0;
    for (int i = 1; i < numVertices; ++i) {
      ScalarType dist = (simplex_.col(idx[i]) - simplex_.col(idx[0]))
                            .template lpNorm<Eigen::Infinity>();
      if (dist > maxDist) {
        maxDist = dist;
      }
    }
    if (maxDist < degenerate_tol_) {
      simplex_ = makeInitialSimplex(simplex_.col(idx[0]));
      // Recompute function values and re-sort indices.
      for (int i = 0; i < numVertices; ++i) {
        f[i] = function(simplex_.col(i));
        idx[i] = i;
      }
      std::sort(idx.begin(), idx.end(),
                [&](int a, int b) { return f[a] < f[b]; });
    }

    // Compute the centroid of the best DIM vertices (all except the worst).
    VectorType x_bar = VectorType::Zero(DIM);
    for (size_t i = 0; i < DIM; ++i) {
      x_bar += simplex_.col(idx[i]);
    }
    x_bar /= static_cast<ScalarType>(DIM);

    // Reflection: compute the reflected point.
    VectorType x_r = (1 + rho_) * x_bar - rho_ * simplex_.col(idx[DIM]);
    if (isCoincident(x_r, x_bar) || isCoincident(x_r, simplex_.col(idx[DIM]))) {
      shrink(simplex_, idx, f, function);
      return StateType(simplex_.col(idx[0]));
    }
    ScalarType f_r = function(x_r);

    if (f_r < f[idx[0]]) {
      // Expansion: if the reflection is the best so far, try to expand further.
      VectorType x_e =
          (1 + rho_ * xi_) * x_bar - rho_ * xi_ * simplex_.col(idx[DIM]);
      ScalarType f_e = function(x_e);
      if (f_e < f_r) {
        simplex_.col(idx[DIM]) = x_e;
      } else {
        simplex_.col(idx[DIM]) = x_r;
      }
    } else if (f_r < f[idx[DIM - 1]]) {
      // Accept the reflected point.
      simplex_.col(idx[DIM]) = x_r;
    } else {
      // Contraction:
      if (f_r < f[idx[DIM]]) {
        // Outside contraction.
        VectorType x_c = (1 + rho_ * gamma_) * x_bar -
                         rho_ * gamma_ * simplex_.col(idx[DIM]);
        ScalarType f_c = function(x_c);
        if (f_c <= f_r) {
          simplex_.col(idx[DIM]) = x_c;
        } else {
          shrink(simplex_, idx, f, function);
        }
      } else {
        // Inside contraction.
        VectorType x_c = (1 - gamma_) * x_bar + gamma_ * simplex_.col(idx[DIM]);
        ScalarType f_c = function(x_c);
        if (f_c < f[idx[DIM]]) {
          simplex_.col(idx[DIM]) = x_c;
        } else {
          shrink(simplex_, idx, f, function);
        }
      }
    }

    // Return the current best vertex.
    return StateType(simplex_.col(idx[0]));
  }

 private:
  MatrixType simplex_;

  // Create an initial simplex given the starting point x using adaptive
  // perturbations.
  MatrixType makeInitialSimplex(const VectorType &x) {
    const size_t DIM = x.rows();
    MatrixType s = MatrixType::Zero(DIM, DIM + 1);
    for (size_t c = 0; c < DIM + 1; ++c) {
      for (size_t r = 0; r < DIM; ++r) {
        s(r, c) = x[r];
        if (r == c - 1) {
          // Use an adaptive perturbation for each coordinate.
          const ScalarType delta =
              (std::abs(x[r]) > 1e-6) ? 0.05 * std::abs(x[r]) : 0.001;
          s(r, c) += delta;
        }
      }
    }
    return s;
  }

  // Returns true if the two vectors are nearly equal (using the infinity norm).
  bool isCoincident(const VectorType &a, const VectorType &b) const {
    return (a - b).template lpNorm<Eigen::Infinity>() < degenerate_tol_;
  }

  // Shrink the simplex toward the best vertex.
  void shrink(MatrixType &s, std::vector<int> &idx, std::vector<ScalarType> &f,
              const FunctionType &function) {
    const size_t DIM = s.rows();
    // Re-evaluate the best vertex.
    f[idx[0]] = function(s.col(idx[0]));
    for (size_t i = 1; i < DIM + 1; ++i) {
      s.col(idx[i]) = sigma_ * s.col(idx[i]) + (1 - sigma_) * s.col(idx[0]);
      f[idx[i]] = function(s.col(idx[i]));
    }
  }
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NELDER_MEAD_H_
