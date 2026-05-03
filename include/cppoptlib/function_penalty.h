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
#ifndef INCLUDE_CPPOPTLIB_FUNCTION_PENALTY_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_PENALTY_H_

#include <Eigen/Core>
#include <type_traits>
#include <vector>

#include "function_problem.h"

namespace cppoptlib::function {

//-------------------------------------------------------------
// Equality penalty: P(x) = 0.5 * [f(x) - g(x)]^2
template <typename F>
auto QuadraticEqualityPenalty(const F& c) {
  return 0.5 * (c * c);
}

//-------------------------------------------------------------
// Inequality penalty for constraint c(x) >= 0:
// P(x) = 0.5 * [min{0, c(x)}]^2. When c(x) >= 0 the penalty is zero.
template <typename F>
auto QuadraticInequalityPenaltyGe(const F& c) {
  auto minExpr = MinZeroExpression<F>(c);
  return 0.5 * (minExpr * minExpr);
}

//-------------------------------------------------------------
// Inequality penalty for constraint c(x) < 0:
// P(x) = 0.5 * [max{0, c(x)}]^2. When c(x) < 0 the penalty is zero.
template <typename F>
auto QuadraticInequalityPenaltyLt(const F& c) {
  auto maxExpr = MaxZeroExpression<F>(c);
  return 0.5 * (maxExpr * maxExpr);
}

// State for Lagrange multipliers (one per constraint type).
template <typename TScalar>
struct LagrangeMultiplierState {
  std::vector<TScalar> equality_multipliers;    // For equality constraints.
  std::vector<TScalar> inequality_multipliers;  // For inequality constraints.

  // Constructor: initializes multipliers with zeros.
  LagrangeMultiplierState(size_t num_eq = 0, size_t num_ineq = 0,
                          TScalar value = TScalar{0})
      : equality_multipliers(num_eq, value),
        inequality_multipliers(num_ineq, value) {}

  LagrangeMultiplierState(std::initializer_list<TScalar> eq,
                          std::initializer_list<TScalar> ineq)
      : equality_multipliers(eq), inequality_multipliers(ineq) {}
};

// State for a single penalty scalar.
template <typename TScalar>
struct PenaltyState {
  TScalar penalty;

  // Constructor: initializes the penalty value.
  explicit PenaltyState(TScalar pen = TScalar(0)) : penalty(pen) {}
};

// --- Forming the Lagrangian Part (without the objective) ---
//
// For equalities the Lagrangian part is `sum_i lambda_i c_i(x)`.  We
// split it out so it can be shared between the composite and any
// diagnostic evaluation.  Inequality constraints do NOT contribute
// here -- their Lagrangian and penalty are folded into a single
// Powell-Hestenes-Rockafellar term by `FormInequalityPart` below
// so that the composite stays bounded below as `g(x)` grows.
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
FunctionExpr<TScalar, Mode, TDim> FormLagrangianPart(
    const ConstrainedOptimizationProblem<TScalar, Mode, TDim>& prob,
    const LagrangeMultiplierState<TScalar>& mult_state) {
  FunctionExpr<TScalar, Mode, TDim> lagrangianPart =
      ConstExpression<TScalar, Mode, TDim>(0);
  for (size_t i = 0; i < prob.equality_constraints.size(); i++) {
    lagrangianPart = lagrangianPart + mult_state.equality_multipliers[i] *
                                          prob.equality_constraints[i];
  }
  return lagrangianPart;
}

// --- Forming the Penalty Part for equalities only -----------------
//
// Equality penalty is the textbook `0.5 * rho * sum_i c_i(x)^2`.
// The inequality penalty is absorbed into `FormInequalityPart` in the
// Powell-Hestenes-Rockafellar formulation.
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
FunctionExpr<TScalar, Mode, TDim> FormPenaltyPart(
    const ConstrainedOptimizationProblem<TScalar, Mode, TDim>& prob,
    const PenaltyState<TScalar>& pen_state) {
  FunctionExpr<TScalar, Mode, TDim> penaltyPart =
      ConstExpression<TScalar, Mode, TDim>(0);
  for (size_t i = 0; i < prob.equality_constraints.size(); i++) {
    const auto& constr = prob.equality_constraints[i];
    auto penaltyExpr = QuadraticEqualityPenalty(constr);
    penaltyPart = penaltyPart + pen_state.penalty * penaltyExpr;
  }
  return penaltyPart;
}

// --- Forming the combined inequality contribution (PHR) ------------
//
// For each inequality `g_j(x) >= 0` with multiplier `mu_j >= 0` and
// penalty `rho`, the Powell-Hestenes-Rockafellar contribution is
//
//     I_j(x) = (1 / (2 rho)) * [ max(0, mu_j - rho * g_j(x))^2 - mu_j^2 ].
//
// The key property of this form -- distinct from the naive
// `- mu * g + 0.5 rho * min(0, g)^2` composition -- is that on the
// strictly-inactive side (g_j > mu_j / rho) the term reduces to the
// *constant* `- mu_j^2 / (2 rho)` with zero gradient.  The composite
// therefore has no "reward for diving deeper into the feasible
// region", which is what makes a naive composition unbounded below
// along rays where a non-convex objective happens to decrease with
// increasing constraint slack.
//
// When `g_j = mu_j / rho` (the switching surface) the expression is
// C^1 across the switch:
//   On the active side:  I_j = -g mu + 0.5 rho g^2 - 0.5 mu^2/rho,
//     differentiated in g gives -mu + rho g, which is 0 at g = mu/rho.
//   On the inactive side: constant, gradient 0.
// Both sides agree at g = mu/rho.
//
// The expression templates carry this piecewise evaluation through
// to gradients automatically via `MaxZeroExpression`.
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
FunctionExpr<TScalar, Mode, TDim> FormInequalityPart(
    const ConstrainedOptimizationProblem<TScalar, Mode, TDim>& prob,
    const LagrangeMultiplierState<TScalar>& mult_state,
    const PenaltyState<TScalar>& pen_state) {
  FunctionExpr<TScalar, Mode, TDim> inequalityPart =
      ConstExpression<TScalar, Mode, TDim>(0);
  const TScalar penalty = pen_state.penalty;
  // A zero or negative penalty in the PHR formulation is ill-defined
  // (we would divide by zero in the constant offset).  The outer
  // loop's auto-scaling guarantees `rho > 0` before this function is
  // called; this guard returns the zero constant if a user bypasses
  // auto-scaling with `rho = 0`.
  if (penalty <= TScalar{0}) {
    return inequalityPart;
  }
  for (size_t j = 0; j < prob.inequality_constraints.size(); j++) {
    const TScalar mu = mult_state.inequality_multipliers[j];
    const auto& g = prob.inequality_constraints[j];
    // argument(x) = mu - penalty * g(x).  At the inner solver's
    // tape level this is a scalar - scalar*Function composition
    // built out of expression templates; `MaxZeroExpression` then
    // clamps the result at zero and carries the piecewise gradient.
    auto argument = mu - penalty * g;
    auto positive_part = MaxZeroExpression<decltype(argument)>(argument);
    // half_inv_rho = 1 / (2 rho).  Folded as a scalar multiplier so
    // the expression template still produces a single weighted
    // squared term per constraint.
    const TScalar half_inv_rho = TScalar{1} / (TScalar{2} * penalty);
    inequalityPart =
        inequalityPart + half_inv_rho * (positive_part * positive_part);
    // Subtract the constant offset `mu^2 / (2 rho)`.  A constant in
    // the composite does not affect the inner solver's minimum but
    // makes the reported composite value match the closed-form PHR
    // expression (useful for diagnostics and tests).
    const TScalar constant_offset = mu * mu * half_inv_rho;
    inequalityPart =
        inequalityPart - ConstExpression<TScalar, Mode, TDim>(constant_offset);
  }
  return inequalityPart;
}

// --- Forming the Penalty-only composite (no Lagrangian part) ---
//
// `L_pen(x) = f(x) + 0.5 rho sum c_eq^2 + 0.5 rho sum min(0, g)^2`.
// This predates the PHR switch in `FormInequalityPart` and keeps the
// naive truncated-quadratic form because the penalty-only composite
// is used for *penalty-method* experiments (pure penalty, no
// multipliers).  It is NOT used by `AugmentedLagrangian`.
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
FunctionExpr<TScalar, Mode, TDim> ToPenalty(
    const ConstrainedOptimizationProblem<TScalar, Mode, TDim>& prob,
    const PenaltyState<TScalar>& pen_state) {
  FunctionExpr<TScalar, Mode, TDim> penaltyPart =
      ConstExpression<TScalar, Mode, TDim>(0);
  for (size_t i = 0; i < prob.equality_constraints.size(); i++) {
    const auto& constr = prob.equality_constraints[i];
    penaltyPart =
        penaltyPart + pen_state.penalty * QuadraticEqualityPenalty(constr);
  }
  for (size_t i = 0; i < prob.inequality_constraints.size(); i++) {
    const auto& constr = prob.inequality_constraints[i];
    penaltyPart =
        penaltyPart + pen_state.penalty * QuadraticInequalityPenaltyGe(constr);
  }
  return prob.objective + penaltyPart;
}
// --- Forming the Full Augmented Lagrangian ---
//
// L_aug(x) = f(x) + LagrangianPart(x) + EqualityPenaltyPart(x)
//          + InequalityPart_PHR(x)
//
// Where:
//   - `LagrangianPart` sums `lambda_i c_i(x)` for equalities.
//   - `FormPenaltyPart` sums `0.5 rho c_i^2` for equalities.
//   - `FormInequalityPart` uses the Powell-Hestenes-Rockafellar
//     formulation for inequalities; see the comment at its
//     definition for the exact expression.
//
// The PHR form is what makes the composite bounded below on problems
// with non-convex objectives and inactive inequalities.  A
// `-mu * g(x)` Lagrangian term in isolation would pull the composite
// to -infinity as g -> +infinity; the PHR form switches to a
// constant there.
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
FunctionExpr<TScalar, Mode, TDim> ToAugmentedLagrangian(
    const ConstrainedOptimizationProblem<TScalar, Mode, TDim>& prob,
    const LagrangeMultiplierState<TScalar>& mult_state,
    const PenaltyState<TScalar>& pen_state) {
  return prob.objective + FormLagrangianPart(prob, mult_state) +
         FormPenaltyPart(prob, pen_state) +
         FormInequalityPart(prob, mult_state, pen_state);
}

}  // namespace cppoptlib::function

#endif  //  INCLUDE_CPPOPTLIB_FUNCTION_PENALTY_H_
