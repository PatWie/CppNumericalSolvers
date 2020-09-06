// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_

#include "../linesearch/armijo.h"
#include "Eigen/Dense"
#include "solver.h"

namespace cppoptlib {

namespace solver {
template <typename function_t>
class NewtonDescent : public Solver<function_t> {
 private:
  using Superclass = Solver<function_t>;
  using state_t = typename Superclass::state_t;

  using scalar_t = typename function_t::scalar_t;
  using hessian_t = typename function_t::hessian_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;
  using function_state_t = typename function_t::state_t;

 public:
  int Order() const override { return 2; }

  void InitializeSolver(const function_state_t &initial_state) override {
    dim_ = initial_state.x.rows();
  }

  function_state_t OptimizationStep(const function_t &function,
                                    const function_state_t &current,
                                    const state_t & /*state*/) override {
    function_state_t next = current;

    constexpr scalar_t safe_guard = 1e-5;
    const hessian_t hessian =
        next.hessian + safe_guard * hessian_t::Identity(dim_, dim_);

    const vector_t delta_x = hessian.lu().solve(-next.gradient);
    const scalar_t rate =
        linesearch::Armijo<function_t, 2>::Search(next.x, delta_x, function);

    return function.Eval(next.x + rate * delta_x, 2);
  }

 private:
  int dim_;
};

}  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
