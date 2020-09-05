// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_

#include "../linesearch/armijo.h"
#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib {

namespace solver {
template <typename function_t>
class ConjugatedGradientDescent : public Solver<function_t, 1> {
 public:
  using Superclass = Solver<function_t, 1>;
  using typename Superclass::state_t;
  using typename Superclass::scalar_t;
  using typename Superclass::vector_t;
  using typename Superclass::function_state_t;

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

    function_state_t next = current;

    const scalar_t rate = linesearch::Armijo<function_t, 1>::Search(
        next.x, search_direction_, function);

    next.x = next.x + rate * search_direction_;
    next.value = function(next.x);
    function.Gradient(next.x, &next.gradient);

    return next;
  }

 private:
  function_state_t previous_;
  vector_t search_direction_;
};

}  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_