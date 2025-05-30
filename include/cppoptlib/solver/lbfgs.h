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
#ifndef INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
#define INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_

#include <algorithm>
#include <cmath>
#include <utility>

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h" // NOLINT

namespace cppoptlib::solver {

template <typename FunctionType, int m = 10>
class Lbfgs
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(
      FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::First ||
          FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::Second,
      "L-BFGS only supports first- or second-order differentiable functions");

private:
  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using progress_t = typename Superclass::progress_t;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using MatrixType = typename FunctionType::MatrixType;

  // Storage for the correction pairs using Eigen matrices.
  using memory_MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, m>;
  using memory_VectorType = Eigen::Matrix<ScalarType, 1, m>;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Superclass::Superclass;

  void InitializeSolver(const FunctionType & /*function*/,
                        const StateType &initial_state) override {
    const size_t dim = initial_state.x.rows();
    x_diff_memory_ = memory_MatrixType::Zero(dim, m);
    grad_diff_memory_ = memory_MatrixType::Zero(dim, m);
    alpha.resize(m);
    // Reset the circular buffer:
    mem_count_ = 0;
    mem_pos_ = 0;
    scaling_factor_ = 1;
  }

  StateType OptimizationStep(const FunctionType &function,
                             const StateType &current,
                             const progress_t & /*progress*/) override {
    constexpr ScalarType eps = std::numeric_limits<ScalarType>::epsilon();
    const ScalarType relative_eps =
        static_cast<ScalarType>(eps) *
        std::max<ScalarType>(ScalarType{1.0}, current.x.norm());

    // --- Preconditioning ---
    // If second-order information is available, use a diagonal preconditioner.
    VectorType precond = VectorType::Ones(current.x.size());
    VectorType current_gradient;
    if constexpr (FunctionType::Differentiability ==
                  cppoptlib::function::DifferentiabilityMode::Second) {
      MatrixType current_hessian;
      function(current.x, &current_gradient, &current_hessian);
      precond = current_hessian.diagonal().cwiseAbs().array() + eps;
      precond = precond.cwiseInverse();
    } else {
      function(current.x, &current_gradient);
    }
    // Precondition the gradient.
    VectorType grad_precond = precond.asDiagonal() * current_gradient;

    // --- Two-Loop Recursion ---
    // Start with the preconditioned gradient as the initial search direction.
    VectorType search_direction = grad_precond;

    // Determine the actual number of stored corrections to use
    const int k = static_cast<int>(mem_count_);

    // First loop: computes q = q - alpha_i * y_i
    // Iterates from the newest correction (k-1) to the oldest (k-m_actual)
    // conceptual_idx refers to the chronological order: 0=oldest,
    // num_valid_corrections-1=newest
    for (int i = k - 1; i >= 0; i--) {
      // Compute the index in chronological order.
      // When mem_count_ < m, corrections are stored in order [0 ...
      // mem_count_-1]. When full, they are stored cyclically starting at
      // mem_pos_ (oldest) up to (mem_pos_ + m - 1) mod m.
      const int idx = (mem_count_ < m) ? i : (mem_pos_ + i) % m;

      const VectorType &s_col = x_diff_memory_.col(idx);
      const VectorType &y_col = grad_diff_memory_.col(idx);

      const ScalarType s_dot_y = s_col.dot(y_col);
      if (std::abs(s_dot_y) < eps) { // Avoid division by zero or near-zero
        continue;
      }
      const ScalarType rho_val = static_cast<ScalarType>(1.0) / s_dot_y;
      alpha(i) = rho_val * s_col.dot(search_direction);
      search_direction -= alpha(i) * y_col;
    }

    // Apply the initial Hessian approximation H_k^0 = gamma_k * I
    // gamma_k = s_{k-1}^T y_{k-1} / (y_{k-1}^T y_{k-1})
    // Here, scaling_factor_ is this gamma_k from the *previous* iteration.
    search_direction *= scaling_factor_;

    // Second loop: computes r = r + s_i * (alpha_i - beta_i)
    // Iterates from the oldest correction (k-m_actual) to the newest (k-1)
    for (int i = 0; i < k; i++) {
      const int idx = (mem_count_ < m) ? i : (mem_pos_ + i) % m;

      const VectorType &s_col = x_diff_memory_.col(idx);
      const VectorType &y_col = grad_diff_memory_.col(idx);

      const ScalarType s_dot_y = s_col.dot(y_col);
      if (std::abs(s_dot_y) < eps) {
        continue;
      }
      const ScalarType rho_val = static_cast<ScalarType>(1.0) / s_dot_y;
      const ScalarType beta = rho_val * y_col.dot(search_direction);
      search_direction += s_col * (alpha(i) - beta);
    }

    // Check descent direction validity.
    ScalarType descent_direction = -current_gradient.dot(search_direction);
    ScalarType alpha_init =
        (current_gradient.norm() > eps) ? 1.0 / current_gradient.norm() : 1.0;
    if (!std::isfinite(descent_direction) ||
        descent_direction > -eps * relative_eps) {
      // Fall back to steepest descent if necessary.
      search_direction = -current_gradient;
      // Reset the correction history if the descent is invalid.
      mem_count_ = 0;
      mem_pos_ = 0;
      alpha_init = 1.0;
    }

    // Perform a line search.
    const ScalarType rate = linesearch::MoreThuente<FunctionType, 1>::Search(
        current.x, -search_direction, function, alpha_init);

    const StateType next = StateType(current.x - rate * search_direction);
    VectorType next_gradient;
    function(next.x, &next_gradient);

    // Compute the differences for the new correction pair.
    const VectorType x_diff = next.x - current.x;
    const VectorType grad_diff = next_gradient - current_gradient;

    // --- Curvature Condition Check with Cautious Update ---
    // We require:
    //   x_diff.dot(grad_diff) > ||x_diff||^2 * ||current.gradient|| *
    //   cautious_factor_
    const ScalarType threshold =
        x_diff.squaredNorm() * current_gradient.norm() * cautious_factor_;
    if (x_diff.dot(grad_diff) > threshold) {
      // Add the new correction pair into the circular buffer.
      if (mem_count_ < static_cast<size_t>(m)) {
        // Still have free space.
        x_diff_memory_.col(mem_count_) = x_diff;
        grad_diff_memory_.col(mem_count_) = grad_diff;
        mem_count_++;
      } else {
        // Buffer full; overwrite the oldest correction.
        x_diff_memory_.col(mem_pos_) = x_diff;
        grad_diff_memory_.col(mem_pos_) = grad_diff;
        mem_pos_ = (mem_pos_ + 1) % m;
      }
    }
    // Update the scaling factor (adaptive damping).
    constexpr ScalarType fallback_value = ScalarType(1e7);
    const ScalarType grad_diff_norm_sq = grad_diff.dot(grad_diff);
    if (std::abs(grad_diff_norm_sq) > eps) {
      ScalarType temp_scaling = grad_diff.dot(x_diff) / grad_diff_norm_sq;
      if (!std::isfinite(temp_scaling) ||
          std::abs(temp_scaling) > fallback_value) {
        scaling_factor_ = fallback_value;
      } else {
        scaling_factor_ = std::max(temp_scaling, eps);
      }
    } else {
      scaling_factor_ = fallback_value;
    }

    return next;
  }

private:
  memory_MatrixType x_diff_memory_;
  memory_MatrixType grad_diff_memory_;
  // Circular buffer state:
  size_t mem_count_ = 0; // Number of corrections stored so far (max m).
  size_t mem_pos_ = 0;   // Index of the oldest correction in the buffer.

  memory_VectorType
      alpha; // Storage for the coefficients in the two-loop recursion.
  ScalarType scaling_factor_ = 1;
  // Cautious factor to determine whether to accept a new correction pair.
  // You may want to expose this parameter or adjust its default value.
  ScalarType cautious_factor_ = 1e-6;
};

} // namespace cppoptlib::solver

#endif // INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
