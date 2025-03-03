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
namespace internal {

template <int m, class T>
void ShiftLeft(T *matrix) {
  matrix->leftCols(m - 1) = matrix->rightCols(m - 1).eval();
}

}  // namespace internal

template <typename function_t, int m = 10>
class Lbfgs : public Solver<function_t> {
  static_assert(function_t::DiffLevel ==
                        cppoptlib::function::Differentiability::First ||
                    function_t::DiffLevel ==
                        cppoptlib::function::Differentiability::Second,
                "L-BFGS only supports first- or second-order "
                "differentiable functions");

 private:
  using Superclass = Solver<function_t>;
  using progress_t = typename Superclass::progress_t;
  using state_t = typename function_t::state_t;

  using scalar_t = typename function_t::scalar_t;
  using hessian_t = typename function_t::matrix_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;

  using memory_matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, m>;
  using memory_vector_t = Eigen::Matrix<scalar_t, 1, m>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void InitializeSolver(const state_t &initial_state) override {
    const size_t dim_ = initial_state.x.rows();
    x_diff_memory_ = memory_matrix_t::Zero(dim_, m);
    grad_diff_memory_ = memory_matrix_t::Zero(dim_, m);
    alpha = memory_vector_t::Zero(m);
    memory_idx_ = 0;
    scaling_factor_ = 1;
  }

  state_t OptimizationStep(const function_t &function, const state_t &current,
                           const progress_t &progress) override {
    vector_t search_direction = current.gradient;

    constexpr scalar_t eps = std::numeric_limits<scalar_t>::epsilon();
    const scalar_t relative_eps =
        static_cast<scalar_t>(eps) *
        std::max<scalar_t>(scalar_t{1.0}, current.x.norm());

    // Algorithm 7.4 (L-BFGS two-loop recursion)
    // // Determine how many stored corrections to use (up to m but not more
    // than available).
    int k = 0;
    if (progress.num_iterations > 0) {
      k = std::min<int>(m, memory_idx_ - 1);
    }

    // First loop (backward pass) for the L-BFGS two-loop recursion.
    for (int i = k - 1; i >= 0; i--) {
      // alpha_i <- rho_i*s_i^T*q
      const scalar_t denom =
          x_diff_memory_.col(i).dot(grad_diff_memory_.col(i));
      if (std::abs(denom) < eps) {
        continue;
      }
      const scalar_t rho = 1.0 / denom;
      alpha(i) = rho * x_diff_memory_.col(i).dot(search_direction);
      // q <- q - alpha_i*y_i
      search_direction -= alpha(i) * grad_diff_memory_.col(i);
    }

    // apply initial Hessian approximation: r <- H_k^0*q
    search_direction *= scaling_factor_;

    // Second loop (forward pass).
    for (int i = 0; i < k; i++) {
      // beta <- rho_i * y_i^T * r
      const scalar_t denom =
          x_diff_memory_.col(i).dot(grad_diff_memory_.col(i));
      if (std::abs(denom) < eps) {
        continue;
      }
      const scalar_t rho = 1.0 / denom;
      const scalar_t beta =
          rho * grad_diff_memory_.col(i).dot(search_direction);
      // r <- r + s_i * ( alpha_i - beta)
      search_direction += x_diff_memory_.col(i) * (alpha(i) - beta);
    }

    // stop with result "H_k*f_f'=q"

    // any issues with the descent direction ?
    // Check the descent direction for validity.
    scalar_t descent_direction = -current.gradient.dot(search_direction);
    scalar_t alpha_init =
        (current.gradient.norm() > eps) ? 1.0 / current.gradient.norm() : 1.0;
    if (!std::isfinite(descent_direction) ||
        descent_direction > -eps * relative_eps) {
      // If the descent direction is invalid or not a descent, revert to
      // steepest descent.
      search_direction = -current.gradient.eval();
      memory_idx_ = 0;
      alpha_init = 1.0;
    }

    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
        current.x, -search_direction, function, alpha_init);

    const state_t next = function.GetState(current.x - rate * search_direction);

    const vector_t x_diff = next.x - current.x;
    const vector_t grad_diff = next.gradient - current.gradient;

    // Update the history
    if (x_diff.dot(grad_diff) > eps * grad_diff.squaredNorm()) {
      if (memory_idx_ < m) {
        x_diff_memory_.col(memory_idx_) = x_diff.eval();
        grad_diff_memory_.col(memory_idx_) = grad_diff.eval();
      } else {
        internal::ShiftLeft<m>(&x_diff_memory_);
        internal::ShiftLeft<m>(&grad_diff_memory_);

        x_diff_memory_.rightCols(1) = x_diff;
        grad_diff_memory_.rightCols(1) = grad_diff;
      }

      memory_idx_++;
    }
    // Adaptive damping in Hessian approximation: update the scaling factor.
    constexpr scalar_t fallback_value =
        scalar_t(1e7);  // Fallback value if update is unstable.
    const scalar_t grad_diff_norm_sq = grad_diff.dot(grad_diff);
    if (std::abs(grad_diff_norm_sq) > eps) {
      scalar_t temp_scaling = grad_diff.dot(x_diff) / grad_diff_norm_sq;
      // If temp_scaling is non-finite or excessively large, use fallback.
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
  size_t memory_idx_ = 0;

  memory_vector_t alpha;
  scalar_t scaling_factor_ = 1;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
