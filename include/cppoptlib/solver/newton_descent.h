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
  using typename Superclass::state_t;
  using typename Superclass::scalar_t;
  using typename Superclass::vector_t;
  using typename Superclass::hessian_t;
  using typename Superclass::function_state_t;

  function_state_t OptimizationStep(const function_t &function,
                                     const function_state_t &current,
                                     const state_t &state) override {
    function_state_t next = current;

    constexpr scalar_t safe_guard = 1e-5;
    hessian_t hessian = next.hessian + safe_guard * hessian_t::Identity();

    const vector_t delta_x = hessian.lu().solve(-next.gradient);
    const scalar_t rate =
        linesearch::Armijo<function_t, 2>::Search(next.x, delta_x, function);

    next.x = next.x + rate * delta_x;

    return function.Eval(next.x);
  }
};

}  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
