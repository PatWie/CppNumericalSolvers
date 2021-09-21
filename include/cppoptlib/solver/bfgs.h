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
 private:
  using Superclass = Solver<function_t>;
  using state_t = typename Superclass::state_t;

  using scalar_t = typename function_t::scalar_t;
  using hessian_t = typename function_t::hessian_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;
  using function_state_t = typename function_t::state_t;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit Bfgs(const State<scalar_t> &stopping_state =
                    DefaultStoppingSolverState<scalar_t>(),
                typename Superclass::callback_t step_callback =
                    GetDefaultStepCallback<scalar_t, vector_t, hessian_t>())
      : Solver<function_t>{stopping_state, std::move(step_callback)} {}

  void InitializeSolver(const function_state_t &initial_state) override {
    dim_ = initial_state.x.rows();
    inverse_hessian_ =
        hessian_t::Identity(initial_state.x.rows(), initial_state.x.rows());
  }

  function_state_t OptimizationStep(const function_t &function,
                                    const function_state_t &current,
                                    const state_t & /*state*/) override {
    vector_t search_direction = -inverse_hessian_ * current.gradient;

    // If not positive definit re-initialize Hessian.
    const scalar_t phi = current.gradient.dot(search_direction);
    if ((phi > 0) || std::isnan(phi)) {
      // no, we reset the hessian approximation
      inverse_hessian_ = hessian_t::Identity(dim_, dim_);
      search_direction = -current.gradient;
    }

    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
        current, search_direction, function);

    const function_state_t next =
        function.Eval(current.x + rate * search_direction, 1);

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
  hessian_t inverse_hessian_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_
