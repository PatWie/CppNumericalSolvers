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
  using typename Superclass::scalar_t;
  using typename Superclass::vector_t;
  using typename Superclass::function_state_t;

  function_state_t optimization_step(const function_t &function,
                                     const function_state_t &state) override {
    function_state_t current(state);

    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::search(
        current.x, -current.gradient, function);

    current.x = current.x - rate * current.gradient;
    current.value = function(current.x);
    function.Gradient(current.x, &current.gradient);

    return current;
  }
};

};  // namespace solver
} /* namespace cppoptlib */

#endif  // INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_
