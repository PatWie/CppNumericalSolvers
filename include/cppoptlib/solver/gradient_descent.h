// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib {

namespace solver {
template <typename Function>
class GradientDescent : public Solver<Function, 1> {
 public:
  using Superclass = Solver<Function, 1>;
  using typename Superclass::ScalarT;
  using typename Superclass::VectorT;
  using typename Superclass::FunctionStateT;

  FunctionStateT optimization_step(const Function &function,
                                   const FunctionStateT &state) override {
    FunctionStateT current(state);

    const ScalarT rate = linesearch::MoreThuente<Function, 1>::search(
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
