// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
#define INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_

#include <algorithm>

#include "../linesearch/armijo.h"
#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib {
namespace solver {

namespace internal {
template <int m, class T>
void ShiftLeft(T *matrix) {
  matrix->leftCols(m - 1) = matrix->rightCols(m - 1).eval();
}
};  // namespace internal

template <typename function_t, int m = 10>
class Lbfgs : public Solver<function_t, 1> {
 public:
  using Superclass = Solver<function_t, 1>;
  using typename Superclass::state_t;
  using typename Superclass::scalar_t;
  using typename Superclass::hessian_t;
  using typename Superclass::vector_t;
  using typename Superclass::function_state_t;

  using memory_t = Eigen::Matrix<scalar_t, function_t::Dim, m>;

  void InitializeSolver(const function_state_t &initial_state) override {
    x_diff_memory_ = memory_t::Zero();
    grad_diff_memory_ = memory_t::Zero();
    alpha = vector_t::Zero();
    memory_idx_ = 0;
    scaling_factor_ = 1;
  }

  function_state_t optimization_step(const function_t &function,
                                     const function_state_t &current,
                                     const state_t &state) override {
    vector_t search_direction = current.gradient;

    constexpr scalar_t absolute_eps = 0.0001;
    const scalar_t relative_eps =
        static_cast<scalar_t>(absolute_eps) *
        std::max<scalar_t>(scalar_t{1.0}, current.x.norm());

    // Algorithm 7.4 (L-BFGS two-loop recursion)
    const int k = std::min<int>(m, memory_idx_);

    // for i = k − 1, k − 2, . . . , k − m
    for (int i = k - 1; i >= 0; i--) {
      // alpha_i <- rho_i*s_i^T*q
      const scalar_t rho =
          1.0 / x_diff_memory_.col(i).dot(grad_diff_memory_.col(i));
      alpha(i) = rho * x_diff_memory_.col(i).dot(search_direction);
      // q <- q - alpha_i*y_i
      search_direction -= alpha(i) * grad_diff_memory_.col(i);
    }

    // r <- H_k^0*q
    search_direction = scaling_factor_ * search_direction;
    // for i k − m, k − m + 1, . . . , k − 1
    for (int i = 0; i < k; i++) {
      // beta <- rho_i * y_i^T * r
      const scalar_t rho =
          1.0 / x_diff_memory_.col(i).dot(grad_diff_memory_.col(i));
      const scalar_t beta =
          rho * grad_diff_memory_.col(i).dot(search_direction);
      // r <- r + s_i * ( alpha_i - beta)
      search_direction += x_diff_memory_.col(i) * (alpha(i) - beta);
    }
    // stop with result "H_k*f_f'=q"

    // any issues with the descent direction ?
    scalar_t descent_direction = -current.gradient.dot(search_direction);
    scalar_t alpha_init = 1.0 / current.gradient.norm();
    if (descent_direction > -absolute_eps * relative_eps) {
      search_direction = -1 * current.gradient;
      memory_idx_ = 0;
      alpha_init = 1.0;
    }

    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::search(
        current.x, -search_direction, function, alpha_init);

    function_state_t next = current;
    next.x = next.x - rate * search_direction;
    next.value = function(next.x);
    function.Gradient(next.x, &next.gradient);

    const vector_t x_diff = next.x - current.x;
    const vector_t grad_diff = next.gradient - current.gradient;

    // Update the history
    if (memory_idx_ < m) {
      x_diff_memory_.col(memory_idx_) = x_diff;
      grad_diff_memory_.col(memory_idx_) = grad_diff;
    } else {
      internal::ShiftLeft<m>(&x_diff_memory_);
      internal::ShiftLeft<m>(&grad_diff_memory_);

      x_diff_memory_.rightCols(1) = x_diff;
      grad_diff_memory_.rightCols(1) = grad_diff;
    }

    memory_idx_++;

    // Update the scaling factor.
    scaling_factor_ = grad_diff.dot(x_diff) / grad_diff.dot(grad_diff);

    return next;
  }

 private:
  memory_t x_diff_memory_;
  memory_t grad_diff_memory_;
  size_t memory_idx_;

  vector_t alpha;
  scalar_t scaling_factor_;
};

}  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_