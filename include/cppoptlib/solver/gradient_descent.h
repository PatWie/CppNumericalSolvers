// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib {

namespace solver {
template <typename function_t>
class GradientDescent : public Solver<function_t, 1> {
 public:
  using Superclass = Solver<function_t, 1>;
  using typename Superclass::state_t;
  using typename Superclass::scalar_t;
  using typename Superclass::vector_t;
  using typename Superclass::function_state_t;

  function_state_t OptimizationStep(const function_t &function,
                                     const function_state_t &current,
                                     const state_t &state) override {
    function_state_t next = current;

    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
        next.x, -next.gradient, function);

    next.x = next.x - rate * next.gradient;
    next.value = function(next.x);
    function.Gradient(next.x, &next.gradient);

    return next;
  }
};

}  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_
