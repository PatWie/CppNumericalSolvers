// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_

#include <utility>

#include "../linesearch/armijo.h"
#include "Eigen/Dense"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {

template <typename function_t>
class NewtonDescent : public Solver<function_t> {
  static_assert(function_t::diff_level ==
                    cppoptlib::function::Differentiability::Second,
                "GradientDescent only supports second-order "
                "differentiable functions");

 private:
  using Superclass = Solver<function_t>;
  using progress_t = typename Superclass::progress_t;
  using state_t = typename function_t::state_t;

  using scalar_t = typename function_t::scalar_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void InitializeSolver(const state_t &initial_state) override {
    dim_ = initial_state.x.rows();
  }

  state_t OptimizationStep(const function_t &function, const state_t &current,
                           const progress_t & /*state*/) override {
    constexpr scalar_t safe_guard = 1e-5;
    const matrix_t hessian =
        current.hessian + safe_guard * matrix_t::Identity(dim_, dim_);

    const vector_t delta_x = hessian.lu().solve(-current.gradient);
    const scalar_t rate =
        linesearch::Armijo<function_t, 2>::Search(current, delta_x, function);

    return state_t(function, current.x + rate * delta_x);
  }

 private:
  int dim_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
