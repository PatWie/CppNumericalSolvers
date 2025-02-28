// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_

#include <utility>

#include "../linesearch/armijo.h"
#include "Eigen/Core"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {
template <typename function_t>
class ConjugatedGradientDescent : public Solver<function_t> {
  static_assert(function_t::diff_level ==
                        cppoptlib::function::Differentiability::First ||
                    function_t::diff_level ==
                        cppoptlib::function::Differentiability::Second,
                "GradientDescent only supports first- or second-order "
                "differentiable functions");

 private:
  using Superclass = Solver<function_t>;
  using progress_t = typename Superclass::progress_t;
  using state_t = typename function_t::state_t;

  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  using matrix_t = typename function_t::matrix_t;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void InitializeSolver(const state_t &initial_state) override {
    previous_ = initial_state;
  }

  state_t OptimizationStep(const function_t &function, const state_t &current,
                           const progress_t &progress) override {
    if (progress.num_iterations == 0) {
      search_direction_ = -current.gradient;
    } else {
      const double beta = current.gradient.dot(current.gradient) /
                          (previous_.gradient.dot(previous_.gradient));
      search_direction_ = -current.gradient + beta * search_direction_;
    }
    previous_ = current;

    const scalar_t rate = linesearch::Armijo<function_t, 1>::Search(
        current, search_direction_, function);

    return state_t(function(current.x + rate * search_direction_));
  }

 private:
  state_t previous_;
  vector_t search_direction_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_
