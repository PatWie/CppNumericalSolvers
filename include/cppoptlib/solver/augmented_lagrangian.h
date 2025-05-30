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

template <typename TScalar, int TDimension = Eigen::Dynamic>
struct AugmentedLagrangeState {
  static constexpr int NumConstraints = 1;
  using VectorType = Eigen::Matrix<TScalar, TDimension, 1>;
  VectorType x;
  TScalar max_violation;

  // State for Lagrange multipliers.
  cppoptlib::function::LagrangeMultiplierState<TScalar> multiplier_state;

  // State for the penalty parameter.
  cppoptlib::function::PenaltyState<TScalar> penalty_state;

  // Constructor #1: Construct from an initial guess, custom initializer lists
  // for equality and inequality multipliers, and a penalty value.
  //
  // Usage:
  //   AugmentedLagrangeState<double, 2> state(x, {0.0}, {0.0}, 1.0);
  AugmentedLagrangeState(const VectorType &init_x,
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
  AugmentedLagrangeState(const VectorType &init_x, size_t num_eq,
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

 private:
  using StateType = AugmentedLagrangeState<typename ProblemType::ScalarType,
                                           ProblemType::Dimension>;
  using Superclass = Solver<ProblemType, StateType>;
  using ProgressType = typename Superclass::ProgressType;

  using ScalarType = typename ProblemType::ScalarType;
  using VectorType = typename ProblemType::VectorType;
  using MatrixType = typename ProblemType::MatrixType;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AugmentedLagrangian(const solver_t &unconstrained_solver)
      : unconstrained_solver_(unconstrained_solver) {}

  AugmentedLagrangian(const ProblemType & /*problem*/,
                      const solver_t &unconstrained_solver)
      : unconstrained_solver_(unconstrained_solver) {}

  void InitializeSolver(const ProblemType & /*function*/,
                        const StateType & /*initial_state*/) override {}

  StateType OptimizationStep(const ProblemType &function,
                             const StateType &state,
                             const ProgressType & /*progress*/) override {
    const auto unconstrained_function =
        cppoptlib::function::ToAugmentedLagrangian(
            function, state.multiplier_state, state.penalty_state);
    const auto [solved_inner_state, inner_progress] =
        unconstrained_solver_.Minimize(
            unconstrained_function,
            cppoptlib::function::FunctionState<ScalarType,
                                               ProblemType::Dimension>(
                state.x));
    StateType next_state = StateType(state);
    next_state.x = solved_inner_state.x;

    float max_violation = 0.0f;

    // Lambda to update multipliers for a set of constraints.
    auto updateMultipliers = [&](const auto &constraints, auto &multipliers,
                                 auto penaltyFn) {
      for (size_t i = 0; i < constraints.size(); ++i) {
        const auto &constr = constraints[i];
        auto penaltyExpr = penaltyFn(constr);
        ScalarType violation = penaltyExpr(next_state.x);
        max_violation = std::max<ScalarType>(max_violation, violation);
        multipliers[i] += next_state.penalty_state.penalty * violation;
      }
    };

    // Update multipliers for equality constraints.
    updateMultipliers(
        function.equality_constraints,
        next_state.multiplier_state.equality_multipliers,
        [](const auto &c) { return quadraticEqualityPenalty(c); });

    // Update multipliers for inequality constraints.
    updateMultipliers(
        function.inequality_constraints,
        next_state.multiplier_state.inequality_multipliers,
        [](const auto &c) { return quadraticInequalityPenalty_ge(c); });

    next_state.penalty_state.penalty = state.penalty_state.penalty * 10;
    next_state.max_violation = max_violation;
    return next_state;
  }

 private:
  solver_t unconstrained_solver_;
};

template <typename ProblemType, typename solver_t>
AugmentedLagrangian(const ProblemType &, const solver_t &)
    -> AugmentedLagrangian<ProblemType, solver_t>;

}  // namespace cppoptlib::solver

#endif  //  INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_
