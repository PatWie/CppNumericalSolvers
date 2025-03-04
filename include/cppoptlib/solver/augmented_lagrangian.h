// Copyright 2025, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_
#define INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_

#include <utility>

#include "../constrained_function.h"  // NOLINT
#include "Eigen/Core"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {

template <typename function_t, typename solver_t>
class AugmentedLagrangian : public Solver<function_t> {
  static_assert(function_t::DiffLevel ==
                        cppoptlib::function::Differentiability::First ||
                    function_t::DiffLevel ==
                        cppoptlib::function::Differentiability::Second,
                "AugmentedLagrangian only supports first- or second-order "
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

  AugmentedLagrangian(const solver_t &inner_solver)
      : inner_solver_(inner_solver) {}

  void InitializeSolver(const function_t & /*function*/,
                        const state_t & /*initial_state*/) override {}

  state_t OptimizationStep(const function_t &function, const state_t &state,
                           const progress_t & /*progress*/) override {
    cppoptlib::function::UnconstrainedFunctionAdapter<function_t>
        unconstrained_function(function, state);

    const auto inner_state = state.AsUnconstrained();
    const auto [solved_inner_state, inner_progress] =
        inner_solver_.Minimize(unconstrained_function, inner_state);

    state_t next_state = function.GetState(
        solved_inner_state.x, state.lagrange_multipliers, state.penalty);

    float max_violation = 0.0f;
    for (std::size_t i = 0; i < function_t::NumConstraints; ++i) {
      const scalar_t violation = next_state.violations[i];
      max_violation = (violation > max_violation) ? violation : max_violation;
      next_state.lagrange_multipliers[i] += next_state.penalty * violation;
    }

    next_state.penalty = state.penalty * 10;
    return next_state;
  }

 private:
  solver_t inner_solver_;
};

}  // namespace cppoptlib::solver

#endif  //  INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_
