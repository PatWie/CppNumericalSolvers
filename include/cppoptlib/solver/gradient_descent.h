// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_

#include <utility>

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {
template <typename function_t>
class GradientDescent : public Solver<function_t> {
  static_assert(function_t::DiffLevel ==
                        cppoptlib::function::Differentiability::First ||
                    function_t::DiffLevel ==
                        cppoptlib::function::Differentiability::Second,
                "GradientDescent only supports first- or second-order "
                "differentiable functions");

 private:
  using Superclass = Solver<function_t>;
  using progress_t = typename Superclass::progress_t;
  using state_t = typename function_t::state_t;

  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void InitializeSolver(const function_t & /*function*/,
                        const state_t & /*initial_state*/) override {}

  state_t OptimizationStep(const function_t &function, const state_t &current,
                           const progress_t & /*progress*/) override {
    vector_t gradient;
    function(current.x, &gradient);
    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
        current.x, -gradient, function);

    return function.GetState(current.x - rate * gradient);
  }
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_
