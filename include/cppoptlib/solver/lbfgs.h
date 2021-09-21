// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
#define INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_

#include <algorithm>
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
 private:
  using Superclass = Solver<function_t>;
  using state_t = typename Superclass::state_t;

  using scalar_t = typename function_t::scalar_t;
  using hessian_t = typename function_t::hessian_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;
  using function_state_t = typename function_t::state_t;

  using memory_matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, m>;
  using memory_vector_t = Eigen::Matrix<scalar_t, 1, m>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit Lbfgs(const State<scalar_t> &stopping_state =
                     DefaultStoppingSolverState<scalar_t>(),
                 typename Superclass::callback_t step_callback =
                     GetDefaultStepCallback<scalar_t, vector_t, hessian_t>())
      : Solver<function_t>{stopping_state, std::move(step_callback)} {}

  void InitializeSolver(const function_state_t &initial_state) override {
    dim_ = initial_state.x.rows();
    x_diff_memory_ = memory_matrix_t::Zero(dim_, m);
    grad_diff_memory_ = memory_matrix_t::Zero(dim_, m);
    alpha = memory_vector_t::Zero(m);
    memory_idx_ = 0;
    scaling_factor_ = 1;
  }

  function_state_t OptimizationStep(const function_t &function,
                                    const function_state_t &current,
                                    const state_t &state) override {
    vector_t search_direction = current.gradient;

    constexpr scalar_t absolute_eps = 0.0001;
    const scalar_t relative_eps =
        static_cast<scalar_t>(absolute_eps) *
        std::max<scalar_t>(scalar_t{1.0}, current.x.norm());

    // Algorithm 7.4 (L-BFGS two-loop recursion)
    int k = 0;
    if (state.num_iterations > 0) {
      k = std::min<int>(m, memory_idx_ - 1);
    }

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
      search_direction = -current.gradient.eval();
      memory_idx_ = 0;
      alpha_init = 1.0;
    }

    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
        current, -search_direction, function, alpha_init);

    const function_state_t next =
        function.Eval(current.x - rate * search_direction, 1);

    const vector_t x_diff = next.x - current.x;
    const vector_t grad_diff = next.gradient - current.gradient;

    // Update the history
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

    // Update the scaling factor.
    scaling_factor_ = grad_diff.dot(x_diff) / grad_diff.dot(grad_diff);

    return next;
  }

 private:
  int dim_;
  memory_matrix_t x_diff_memory_;
  memory_matrix_t grad_diff_memory_;
  size_t memory_idx_;

  memory_vector_t alpha;
  scalar_t scaling_factor_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
