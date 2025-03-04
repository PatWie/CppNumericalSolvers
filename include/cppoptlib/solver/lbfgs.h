// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
#define INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_

#include <algorithm>
#include <cmath>
#include <utility>

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {

template <typename function_t, int m = 10>
class Lbfgs : public Solver<function_t> {
  static_assert(
      function_t::DiffLevel == cppoptlib::function::Differentiability::First ||
          function_t::DiffLevel ==
              cppoptlib::function::Differentiability::Second,
      "L-BFGS only supports first- or second-order differentiable functions");

 private:
  using Superclass = Solver<function_t>;
  using progress_t = typename Superclass::progress_t;
  using state_t = typename function_t::state_t;
  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  using matrix_t = typename function_t::matrix_t;

  // Storage for the correction pairs using Eigen matrices.
  using memory_matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, m>;
  using memory_vector_t = Eigen::Matrix<scalar_t, 1, m>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Superclass::Superclass;

  void InitializeSolver(const function_t & /*function*/,
                        const state_t &initial_state) override {
    const size_t dim = initial_state.x.rows();
    x_diff_memory_ = memory_matrix_t::Zero(dim, m);
    grad_diff_memory_ = memory_matrix_t::Zero(dim, m);
    alpha.resize(m);
    // Reset the circular buffer:
    mem_count_ = 0;
    mem_pos_ = 0;
    scaling_factor_ = 1;
  }

  state_t OptimizationStep(const function_t &function, const state_t &current,
                           const progress_t & /*progress*/) override {
    constexpr scalar_t eps = std::numeric_limits<scalar_t>::epsilon();
    const scalar_t relative_eps =
        static_cast<scalar_t>(eps) *
        std::max<scalar_t>(scalar_t{1.0}, current.x.norm());

    // --- Preconditioning ---
    // If second-order information is available, use a diagonal preconditioner.
    vector_t precond = vector_t::Ones(current.x.size());
    vector_t current_gradient;
    if constexpr (function_t::DiffLevel ==
                  cppoptlib::function::Differentiability::Second) {
      matrix_t current_hessian;
      function(current.x, &current_gradient, &current_hessian);
      precond = current_hessian.diagonal().cwiseAbs().array() + eps;
      precond = precond.cwiseInverse();
    } else {
      function(current.x, &current_gradient);
    }
    // Precondition the gradient.
    vector_t grad_precond = precond.asDiagonal() * current_gradient;

    // --- Two-Loop Recursion ---
    // Start with the preconditioned gradient as the initial search direction.
    vector_t search_direction = grad_precond;

    // Determine the number of corrections available for the two-loop recursion.
    // We exclude the most recent correction (which was just computed) from use.
    int k = (mem_count_ > 0 ? static_cast<int>(mem_count_) - 1 : 0);

    // --- First Loop (Backward Pass) ---
    // Iterate over stored corrections in reverse chronological order.
    for (int i = k - 1; i >= 0; i--) {
      // Compute the index in chronological order.
      // When mem_count_ < m, corrections are stored in order [0 ...
      // mem_count_-1]. When full, they are stored cyclically starting at
      // mem_pos_ (oldest) up to (mem_pos_ + m - 1) mod m.
      int idx = (mem_count_ < m ? i : ((mem_pos_ + i) % m));
      const scalar_t denom =
          x_diff_memory_.col(idx).dot(grad_diff_memory_.col(idx));
      if (std::abs(denom) < eps) {
        continue;
      }
      const scalar_t rho = 1.0 / denom;
      alpha(i) = rho * x_diff_memory_.col(idx).dot(search_direction);
      search_direction -= alpha(i) * grad_diff_memory_.col(idx);
    }

    // Apply the initial Hessian approximation.
    search_direction *= scaling_factor_;

    // --- Second Loop (Forward Pass) ---
    for (int i = 0; i < k; i++) {
      int idx = (mem_count_ < m ? i : ((mem_pos_ + i) % m));
      const scalar_t denom =
          x_diff_memory_.col(idx).dot(grad_diff_memory_.col(idx));
      if (std::abs(denom) < eps) {
        continue;
      }
      const scalar_t rho = 1.0 / denom;
      const scalar_t beta =
          rho * grad_diff_memory_.col(idx).dot(search_direction);
      search_direction += x_diff_memory_.col(idx) * (alpha(i) - beta);
    }

    // Check descent direction validity.
    scalar_t descent_direction = -current_gradient.dot(search_direction);
    scalar_t alpha_init =
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
    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
        current.x, -search_direction, function, alpha_init);

    const state_t next = function.GetState(current.x - rate * search_direction);
    vector_t next_gradient;
    function(next.x, &next_gradient);

    // Compute the differences for the new correction pair.
    const vector_t x_diff = next.x - current.x;
    const vector_t grad_diff = next_gradient - current_gradient;

    // --- Curvature Condition Check with Cautious Update ---
    // We require:
    //   x_diff.dot(grad_diff) > ||x_diff||^2 * ||current.gradient|| *
    //   cautious_factor_
    const scalar_t threshold =
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
    constexpr scalar_t fallback_value = scalar_t(1e7);
    const scalar_t grad_diff_norm_sq = grad_diff.dot(grad_diff);
    if (std::abs(grad_diff_norm_sq) > eps) {
      scalar_t temp_scaling = grad_diff.dot(x_diff) / grad_diff_norm_sq;
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
  memory_matrix_t x_diff_memory_;
  memory_matrix_t grad_diff_memory_;
  // Circular buffer state:
  size_t mem_count_ = 0;  // Number of corrections stored so far (max m).
  size_t mem_pos_ = 0;    // Index of the oldest correction in the buffer.

  memory_vector_t
      alpha;  // Storage for the coefficients in the two-loop recursion.
  scalar_t scaling_factor_ = 1;
  // Cautious factor to determine whether to accept a new correction pair.
  // You may want to expose this parameter or adjust its default value.
  scalar_t cautious_factor_ = 1e-6;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
