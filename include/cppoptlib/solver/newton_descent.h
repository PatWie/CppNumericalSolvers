// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_

#include <utility>

#include "../linesearch/armijo.h"
#include "Eigen/Dense"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {

template <typename function_t>
class NewtonDescent : public Solver<function_t, 2> {
 private:
  using Superclass = Solver<function_t, 2>;
  using state_t = typename Superclass::state_t;

  using scalar_t = typename function_t::scalar_t;
  using hessian_t = typename function_t::hessian_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;
  using function_state_t = typename function_t::state_t;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit NewtonDescent(
      const State<scalar_t> &stopping_state =
          DefaultStoppingSolverState<scalar_t>(),
      typename Superclass::callback_t step_callback =
          GetDefaultStepCallback<scalar_t, vector_t, hessian_t>())
      : Superclass{stopping_state, std::move(step_callback)} {}

  void InitializeSolver(const function_state_t &initial_state) override {
    dim_ = initial_state.x.rows();
  }

  function_state_t OptimizationStep(const function_t &function,
                                    const function_state_t &current,
                                    const state_t & /*state*/) override {
    constexpr scalar_t safe_guard = 1e-5;
    const hessian_t hessian =
        *(current.hessian) + safe_guard * hessian_t::Identity(dim_, dim_);

    const vector_t delta_x = hessian.lu().solve(-current.gradient);
    const scalar_t rate =
        linesearch::Armijo<function_t, Superclass::Order>::Search(
            current, delta_x, function);

    return function.Eval(current.x + rate * delta_x, Superclass::Order);
  }

 private:
  int dim_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
