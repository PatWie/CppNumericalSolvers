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
#ifndef INCLUDE_CPPOPTLIB_FUNCTION_PROBLEM_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_PROBLEM_H_

#include <Eigen/Core>
#include <type_traits>
#include <vector>

#include "function_base.h"
#include "function_expressions.h"

namespace cppoptlib::function {

// ConstrainedOptimizationProblem represents a constrained optimization problem:
//   minimize f(x)
//   subject to: c(x) == 0  and  c(x) >= 0
//
// A single `Mode` template parameter describes the differentiability of
// both the objective and the constraints.  Earlier revisions allowed an
// asymmetric `Second` objective with `First` constraints, but no code
// path in the augmented-Lagrangian composite ever pulled on the extra
// Hessian edge -- the inner solver consumes a composite whose mode is
// the minimum of the two and therefore never calls the objective's
// three-argument operator.  The separate `ModeConstr` parameter was
// dead weight that also made CTAD ambiguous on the common case where
// both modes agreed.  Users who want to mix modes can still wrap a
// higher-mode expression into a lower-mode `FunctionExpr` (the
// converting constructor accepts the downgrade).
template <typename TScalar,
          DifferentiabilityMode Mode = DifferentiabilityMode::First,
          int TDimension = Eigen::Dynamic>
struct ConstrainedOptimizationProblem {
  static constexpr int Dimension = TDimension;

  using ScalarType = TScalar;
  using VectorType = Eigen::Matrix<ScalarType, Dimension, 1>;
  using MatrixType = Eigen::Matrix<ScalarType, Dimension, Dimension>;

  using ObjectiveFunctionType = FunctionExpr<TScalar, Mode, TDimension>;
  using ConstraintFunctionType = FunctionExpr<TScalar, Mode, TDimension>;
  static constexpr DifferentiabilityMode Differentiability = Mode;

  const FunctionExpr<TScalar, Mode, TDimension> objective;  // f(x)
  const std::vector<FunctionExpr<TScalar, Mode, TDimension>>
      equality_constraints;  // c(x) == 0
  const std::vector<FunctionExpr<TScalar, Mode, TDimension>>
      inequality_constraints;  // c(x) >= 0

  ConstrainedOptimizationProblem(
      const FunctionExpr<TScalar, Mode, TDimension> obj,
      const std::vector<FunctionExpr<TScalar, Mode, TDimension>>
          eq_constraints = {},
      const std::vector<FunctionExpr<TScalar, Mode, TDimension>>
          ineq_constraints = {})
      : objective(std::move(obj)),
        equality_constraints(std::move(eq_constraints)),
        inequality_constraints(std::move(ineq_constraints)) {}
};

// Deduction guide: one-argument form (no constraints).
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
ConstrainedOptimizationProblem(const FunctionExpr<TScalar, Mode, TDim>&)
    -> ConstrainedOptimizationProblem<TScalar, Mode, TDim>;

// Deduction guide: two-argument form (objective + equality constraints).
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
ConstrainedOptimizationProblem(
    const FunctionExpr<TScalar, Mode, TDim>&,
    std::initializer_list<FunctionExpr<TScalar, Mode, TDim>>)
    -> ConstrainedOptimizationProblem<TScalar, Mode, TDim>;

// Deduction guide: three-argument form (objective + equality + inequality).
template <typename TScalar, DifferentiabilityMode Mode, int TDim>
ConstrainedOptimizationProblem(
    const FunctionExpr<TScalar, Mode, TDim>&,
    std::initializer_list<FunctionExpr<TScalar, Mode, TDim>>,
    std::initializer_list<FunctionExpr<TScalar, Mode, TDim>>)
    -> ConstrainedOptimizationProblem<TScalar, Mode, TDim>;

}  // namespace cppoptlib::function

#endif  //  INCLUDE_CPPOPTLIB_FUNCTION_PROBLEM_H_
