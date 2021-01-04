// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_

#include "../linesearch/armijo.h"
#include "Eigen/Core"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {
template <typename function_t>
class ConjugatedGradientDescent : public Solver<function_t> {
 private:
  using Superclass = Solver<function_t>;
  using state_t = typename Superclass::state_t;

  using scalar_t = typename function_t::scalar_t;
  using hessian_t = typename function_t::hessian_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;
  using function_state_t = typename function_t::state_t;

 public:
  void InitializeSolver(const function_state_t &initial_state) override {
    previous_ = initial_state;
  }

  function_state_t OptimizationStep(const function_t &function,
                                    const function_state_t &current,
                                    const state_t &state) override {
    if (state.num_iterations == 0) {
      search_direction_ = -current.gradient;
    } else {
      const double beta = current.gradient.dot(current.gradient) /
                          (previous_.gradient.dot(previous_.gradient));
      search_direction_ = -current.gradient + beta * search_direction_;
    }
    previous_ = current;

    const scalar_t rate = linesearch::Armijo<function_t, 1>::Search(
        current.x, search_direction_, function);

    return function.Eval(current.x + rate * search_direction_, 1);
  }

 private:
  function_state_t previous_;
  vector_t search_direction_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_
