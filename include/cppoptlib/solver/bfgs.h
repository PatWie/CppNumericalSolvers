// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_
#define INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_

#include <utility>

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {
template <typename function_t>
class Bfgs : public Solver<function_t> {
  static_assert(function_t::DiffLevel ==
                        cppoptlib::function::Differentiability::First ||
                    function_t::DiffLevel ==
                        cppoptlib::function::Differentiability::Second,
                "GradientDescent only supports first- or second-order "
                "differentiable functions");

 private:
  using Superclass = Solver<function_t>;
  using progress_t = typename Superclass::progress_t;
  using state_t = typename function_t::state_t;

  using scalar_t = typename function_t::scalar_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void InitializeSolver(const state_t &initial_state) override {
    dim_ = initial_state.x.rows();
    inverse_hessian_ =
        matrix_t::Identity(initial_state.x.rows(), initial_state.x.rows());
  }

  state_t OptimizationStep(const function_t &function, const state_t &current,
                           const progress_t & /*progress*/) override {
    vector_t search_direction = -inverse_hessian_ * current.gradient;

    // If not positive definit re-initialize Hessian.
    const scalar_t phi = current.gradient.dot(search_direction);
    if ((phi > 0) || std::isnan(phi)) {
      // no, we reset the hessian approximation
      inverse_hessian_ = matrix_t::Identity(dim_, dim_);
      search_direction = -current.gradient;
    }

    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
        current.x, search_direction, function);

    const state_t next = function.GetState(current.x + rate * search_direction);

    // Update inverse Hessian estimate.
    const vector_t s = rate * search_direction;
    const vector_t y = next.gradient - current.gradient;
    const scalar_t rho = 1.0 / y.dot(s);

    inverse_hessian_ =
        inverse_hessian_ -
        rho * (s * (y.transpose() * inverse_hessian_) +
               (inverse_hessian_ * y) * s.transpose()) +
        rho * (rho * y.dot(inverse_hessian_ * y) + 1.0) * (s * s.transpose());

    return next;
  }

 private:
  int dim_;
  matrix_t inverse_hessian_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_
