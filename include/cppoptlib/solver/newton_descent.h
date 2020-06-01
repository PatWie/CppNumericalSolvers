// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_

#include "../linesearch/armijo.h"
#include "Eigen/Dense"
#include "solver.h"

namespace cppoptlib {

namespace solver {
template <typename function_t>
class NewtonDescent : public Solver<function_t, 2> {
 public:
  using Superclass = Solver<function_t, 2>;
  using typename Superclass::Dim;
  using typename Superclass::scalar_t;
  using typename Superclass::vector_t;
  using typename Superclass::hessian_t;
  using typename Superclass::function_state_t;

  function_state_t optimization_step(const function_t &function,
                                     const function_state_t &state) override {
    function_state_t current(state);

    constexpr scalar_t safe_guard = 1e-5;
    current.hessian += safe_guard * hessian_t::Identity();

    const vector_t delta_x = current.hessian.lu().solve(-current.gradient);
    const scalar_t rate =
        linesearch::Armijo<function_t, 2>::search(current.x, delta_x, function);

    current.x = current.x - rate * delta_x;

    return function.Eval(current.x);
  }
};

};  // namespace solver
} /* namespace cppoptlib */

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_