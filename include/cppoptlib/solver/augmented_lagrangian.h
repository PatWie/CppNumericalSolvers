// CPPNumericalSolvers - A lightweight C++ numerical optimization library
// Copyright (c) 2014    Patrick Wieschollek + Contributors
// Licensed under the MIT License (see below).
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Author: Patrick Wieschollek
//
// More details can be found in the project documentation:
// https://github.com/PatWie/CppNumericalSolvers

#ifndef INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_
#define INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_

#include <utility>

#include "../function_penalty.h"
#include "../function_problem.h"
#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib::solver {

// Tunable knobs for the augmented-Lagrangian outer loop.  Defaults match
// Nocedal & Wright §17.4: penalty grows by a factor of 10 when the
// observed constraint violation does not shrink to at most 1/4 of the
// previous iterate's violation.  Users who already have a well-
// conditioned problem can set `penalty_growth_factor = 1` to disable
// penalty growth entirely.
template <typename TScalar>
struct AugmentedLagrangianConfig {
  // Multiplicative factor applied to the penalty `rho` when the most
  // recent outer iteration's `max_violation` did not shrink by at least
  // the ratio below.  Must be `>= 1`; values `> 1` eventually drive
  // feasibility.
  TScalar penalty_growth_factor = TScalar{10};

  // If `max_violation_new <= violation_shrink_ratio * max_violation_prev`
  // then the penalty is left unchanged -- the multipliers alone did
  // enough work on the last iterate.  Textbook value is `0.25`.
  TScalar violation_shrink_ratio = TScalar{0.25};
};

template <typename TScalar, int TDimension = Eigen::Dynamic>
struct AugmentedLagrangeState {
  // Marks this state as a constrained-solver state; `Progress::Update`
  // picks a different convergence branch when it reads this trait.
  static constexpr bool IsConstrained = true;
  using VectorType = Eigen::Matrix<TScalar, TDimension, 1>;
  VectorType x;

  // State for Lagrange multipliers.  Declared BEFORE `max_violation` so
  // the initializer-list order in the constructors below -- which
  // populates multipliers and penalty before the violation counter --
  // matches the declaration order.  The previous declaration order
  // (`max_violation` first) produced a `-Wreorder` warning on every
  // instantiation.
  cppoptlib::function::LagrangeMultiplierState<TScalar> multiplier_state;

  // State for the penalty parameter.
  cppoptlib::function::PenaltyState<TScalar> penalty_state;

  // Largest constraint violation observed at the most recent outer
  // iteration.  For equality `c = 0` this is `|c(x)|`; for inequality
  // `c >= 0` this is `max(0, -c(x))`.  Compared against the solver's
  // `constraint_threshold` stopping tolerance in `Progress::Update`.
  TScalar max_violation;

  // Constructor #1: Construct from an initial guess, custom initializer lists
  // for equality and inequality multipliers, and a penalty value.
  //
  // Usage:
  //   AugmentedLagrangeState<double, 2> state(x, {0.0}, {0.0}, 1.0);
  AugmentedLagrangeState(const VectorType& init_x,
                         std::initializer_list<TScalar> eq_multipliers,
                         std::initializer_list<TScalar> ineq_multipliers,
                         TScalar penalty)
      : x(init_x),
        multiplier_state(eq_multipliers, ineq_multipliers),
        penalty_state(penalty),
        max_violation(0) {}

  // Constructor #2: Construct from an initial guess, numbers of equality and
  // inequality constraints (multipliers will be zero-initialized), and an
  // optional penalty value.
  //
  // Usage:
  //   AugmentedLagrangeState<double, 2> state(x, 1, 1, 1.0);
  AugmentedLagrangeState(const VectorType& init_x, size_t num_eq,
                         size_t num_ineq, TScalar penalty = TScalar(1))
      : x(init_x),
        multiplier_state(num_eq, num_ineq, TScalar(0)),
        penalty_state(penalty),
        max_violation(0) {}
};

template <typename ProblemType, typename solver_t>
class AugmentedLagrangian
    : public Solver<ProblemType,
                    AugmentedLagrangeState<typename ProblemType::ScalarType,
                                           ProblemType::Dimension>> {
  static_assert(ProblemType::Differentiability ==
                        cppoptlib::function::DifferentiabilityMode::First ||
                    ProblemType::Differentiability ==
                        cppoptlib::function::DifferentiabilityMode::Second,
                "AugmentedLagrangian only supports first- or second-order "
                "differentiable functions");

 public:
  using StateType = AugmentedLagrangeState<typename ProblemType::ScalarType,
                                           ProblemType::Dimension>;
  using Superclass = Solver<ProblemType, StateType>;
  using ProgressType = typename Superclass::ProgressType;

  using ScalarType = typename ProblemType::ScalarType;
  using VectorType = typename ProblemType::VectorType;
  using MatrixType = typename ProblemType::MatrixType;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Construct by binding the problem and the inner unconstrained solver
  // at once.  The problem is stored by value; callers can then use the
  // convenience `Minimize(state)` overload and never repeat the problem
  // argument.  Storing by value (rather than by reference) matches the
  // inner solver's storage mode and avoids dangling pointers when users
  // build the solver as a temporary.  Both the problem and the inner
  // solver are copied once here; neither is expected to be expensive --
  // the problem holds type-erased `FunctionExpr`s (shared-pointer-like
  // clone cost) and the inner solver holds only its own scalar state.
  AugmentedLagrangian(const ProblemType& problem,
                      const solver_t& unconstrained_solver,
                      AugmentedLagrangianConfig<ScalarType> config = {})
      : problem_(problem),
        unconstrained_solver_(unconstrained_solver),
        config_(config) {}

  // Bring the base's two-argument `Minimize(problem, state)` overload
  // into scope so that it is not hidden by the one-argument
  // `Minimize(state)` convenience form below.  Users who want to re-run
  // the solver on a different problem can still call the two-argument
  // form.
  using Superclass::Minimize;

  // Convenience: run the solver on the problem bound at construction.
  // Equivalent to `Minimize(stored_problem, state)`.
  std::tuple<StateType, ProgressType> Minimize(const StateType& state) {
    return Minimize(problem_, state);
  }

  void InitializeSolver(const ProblemType& /*function*/,
                        const StateType& /*initial_state*/) override {}

  StateType OptimizationStep(const ProblemType& function,
                             const StateType& state,
                             const ProgressType& /*progress*/) override {
    const auto unconstrained_function =
        cppoptlib::function::ToAugmentedLagrangian(
            function, state.multiplier_state, state.penalty_state);
    // Discard the inner solver's `Progress` here -- the outer loop's
    // `Progress` is what stops the augmented-Lagrangian iteration, and
    // the inner progress is an implementation detail.  Named binding
    // would trigger `-Wunused-variable`.
    const auto solved_inner_state = std::get<0>(unconstrained_solver_.Minimize(
        unconstrained_function,
        cppoptlib::function::FunctionState<ScalarType, ProblemType::Dimension>(
            state.x)));
    StateType next_state = StateType(state);
    next_state.x = solved_inner_state.x;

    // `max_violation` must be `ScalarType` -- using `float` would silently
    // downcast a `double` problem's violation and break the default
    // stopping tolerance of `1e-5`.
    ScalarType max_violation = ScalarType{0};
    const ScalarType penalty = next_state.penalty_state.penalty;

    // Equality constraints: update is the textbook
    //     lambda_{k+1} = lambda_k + rho_k * c(x_k)
    // derived from the augmented-Lagrangian stationarity condition
    //     grad f + sum_i (lambda_i + rho * c_i) grad c_i = 0.
    // Violation is |c(x_k)|; equality is satisfied when c = 0.
    for (size_t i = 0; i < function.equality_constraints.size(); ++i) {
      const ScalarType constraint_value =
          function.equality_constraints[i](next_state.x);
      max_violation =
          std::max<ScalarType>(max_violation, std::abs(constraint_value));
      next_state.multiplier_state.equality_multipliers[i] +=
          penalty * constraint_value;
    }

    // Inequality constraints: the `c(x) >= 0` convention with penalty
    //     P = 0.5 * min(0, c)^2
    // and Lagrangian term `- mu * c` gives the update
    //     mu_{k+1} = max(0, mu_k - rho_k * c(x_k))
    // via the same stationarity identification as above.  The `max(0, .)`
    // projection enforces dual feasibility mu >= 0.  Violation is
    // `max(0, -c(x_k))`: positive c means the constraint is satisfied
    // with slack, so no violation.
    for (size_t i = 0; i < function.inequality_constraints.size(); ++i) {
      const ScalarType constraint_value =
          function.inequality_constraints[i](next_state.x);
      const ScalarType violation =
          std::max<ScalarType>(ScalarType{0}, -constraint_value);
      max_violation = std::max<ScalarType>(max_violation, violation);
      ScalarType& mu = next_state.multiplier_state.inequality_multipliers[i];
      mu = std::max<ScalarType>(ScalarType{0}, mu - penalty * constraint_value);
    }

    // Penalty schedule: only grow `rho` when the observed violation
    // failed to shrink sufficiently since the previous outer iteration.
    // `state.max_violation` holds the violation measured at the *end*
    // of the previous outer iteration (or 0 on the very first call,
    // because `AugmentedLagrangeState` initializes it to 0).  On that
    // first call the test `max_violation > ratio * previous` reduces to
    // `max_violation > 0`, which is true whenever the starting point is
    // infeasible -- exactly the case where growing `rho` is needed.
    const ScalarType previous_max_violation = state.max_violation;
    const bool violation_shrank_enough =
        max_violation <=
        config_.violation_shrink_ratio * previous_max_violation;
    next_state.penalty_state.penalty =
        violation_shrank_enough ? penalty
                                : penalty * config_.penalty_growth_factor;
    next_state.max_violation = max_violation;
    return next_state;
  }

 private:
  ProblemType problem_;
  solver_t unconstrained_solver_;
  AugmentedLagrangianConfig<ScalarType> config_;
};

template <typename ProblemType, typename solver_t>
AugmentedLagrangian(const ProblemType&, const solver_t&)
    -> AugmentedLagrangian<ProblemType, solver_t>;

}  // namespace cppoptlib::solver

#endif  //  INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_
