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
// This function sums the multiplier-weighted constraints:
// LagrangianPart(x) = Σ_i [ multiplier_i * c_i(x) ]
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
FunctionExpr<TScalar, Mode, TDim> FormLagrangianPart(
    const ConstrainedOptimizationProblem<TScalar, Mode, TDim>& prob,
    const LagrangeMultiplierState<TScalar>& mult_state) {
  // Initialize to the zero function.
  FunctionExpr<TScalar, Mode, TDim> lagrangianPart =
      ConstExpression<TScalar, Mode, TDim>(0);

  // Sum contributions from equality constraints.
  for (size_t i = 0; i < prob.equality_constraints.size(); i++) {
    lagrangianPart = lagrangianPart + mult_state.equality_multipliers[i] *
                                          prob.equality_constraints[i];
  }

  // Sum contributions from inequality constraints.  Convention:
  // inequality constraints are `c_i(x) >= 0` and their multipliers
  // `mu_i >= 0`.  The Lagrangian contribution is `- mu_i * c_i(x)` so
  // that at an active KKT point the objective gradient and the
  // multiplier-weighted constraint gradient balance correctly:
  //     grad f = sum_i mu_i * grad c_i .
  // Writing `+ mu_i * c_i(x)` here (as an earlier version did) pushed
  // the inner solver *away* from the feasible boundary and made KKT
  // recovery impossible for any active inequality.
  for (size_t i = 0; i < prob.inequality_constraints.size(); i++) {
    lagrangianPart = lagrangianPart - mult_state.inequality_multipliers[i] *
                                          prob.inequality_constraints[i];
  }

  return lagrangianPart;
}

// --- Forming the Penalty Part (without the objective) ---
// This function sums the penalty terms for all constraints:
// PenaltyPart(x) = Σ_i [ penalty * P_i(x) ]
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
FunctionExpr<TScalar, Mode, TDim> FormPenaltyPart(
    const ConstrainedOptimizationProblem<TScalar, Mode, TDim>& prob,
    const PenaltyState<TScalar>& pen_state) {
  // Initialize to the zero function.
  FunctionExpr<TScalar, Mode, TDim> penaltyPart =
      ConstExpression<TScalar, Mode, TDim>(0);

  // Process equality constraints.
  for (size_t i = 0; i < prob.equality_constraints.size(); i++) {
    const auto& constr = prob.equality_constraints[i];
    auto penaltyExpr = QuadraticEqualityPenalty(constr);
    penaltyPart = penaltyPart + pen_state.penalty * penaltyExpr;
  }

  // Process inequality constraints.
  for (size_t i = 0; i < prob.inequality_constraints.size(); i++) {
    const auto& constr = prob.inequality_constraints[i];
    auto penaltyExpr = QuadraticInequalityPenaltyGe(constr);
    penaltyPart = penaltyPart + pen_state.penalty * penaltyExpr;
  }

  return penaltyPart;
}

// --- Forming the Penalty ---
// L_aug(x) = f(x) + PenaltyPart(x)
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
FunctionExpr<TScalar, Mode, TDim> ToPenalty(
    const ConstrainedOptimizationProblem<TScalar, Mode, TDim>& prob,
    const PenaltyState<TScalar>& pen_state) {
  return prob.objective + FormPenaltyPart(prob, pen_state);
}
// --- Forming the Full Augmented Lagrangian ---
// L_aug(x) = f(x) + LagrangianPart(x) + PenaltyPart(x)
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
FunctionExpr<TScalar, Mode, TDim> ToAugmentedLagrangian(
    const ConstrainedOptimizationProblem<TScalar, Mode, TDim>& prob,
    const LagrangeMultiplierState<TScalar>& mult_state,
    const PenaltyState<TScalar>& pen_state) {
  return prob.objective + FormLagrangianPart(prob, mult_state) +
         FormPenaltyPart(prob, pen_state);
}

}  // namespace cppoptlib::function

#endif  //  INCLUDE_CPPOPTLIB_FUNCTION_PENALTY_H_
