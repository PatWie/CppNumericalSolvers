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
// This file contains implementation details of numerical optimization solvers
// using automatic differentiation and gradient-based methods.
// CPPNumericalSolvers provides a flexible and efficient interface for solving
// unconstrained and constrained optimization problems.
//
// More details can be found in the project documentation:
// https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_FUNCTION_PROBLEM_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_PROBLEM_H_

#include <Eigen/Core>
#include <type_traits>
#include <vector>

#include "function_problem.h"

namespace cppoptlib::function {

// ConstrainedOptimizationProblem represents a constrained optimization problem:
//   minimize f(x)
//   subject to: c(x) == 0  and  c(x) >= 0
//
template <typename TScalar, DifferentiabilityMode ModeObj,
          DifferentiabilityMode ModeConstr, int TDimension>
struct ConstrainedOptimizationProblem {
  static constexpr int Dimension = TDimension;

  using ScalarType = TScalar;
  using VectorType = Eigen::Matrix<ScalarType, Dimension, 1>;
  using MatrixType = Eigen::Matrix<ScalarType, Dimension, Dimension>;

  using ObjectiveFunctionType = FunctionExpr<TScalar, ModeObj, TDimension>;
  using ConstraintFunctionType = FunctionExpr<TScalar, ModeConstr, TDimension>;
  static constexpr DifferentiabilityMode Differentiability =
      MinDifferentiabilityMode<ModeObj, ModeConstr>::value;

  const FunctionExpr<TScalar, ModeObj, TDimension> objective;  // f(x)
  const std::vector<FunctionExpr<TScalar, ModeConstr, TDimension>>
      equality_constraints;  // c(x) == 0
  const std::vector<FunctionExpr<TScalar, ModeConstr, TDimension>>
      inequality_constraints;  // c(x) >= 0

  ConstrainedOptimizationProblem(
      const FunctionExpr<TScalar, ModeObj, TDimension> obj,
      const std::vector<FunctionExpr<TScalar, ModeConstr, TDimension>>
          eq_constraints = {},
      const std::vector<FunctionExpr<TScalar, ModeConstr, TDimension>>
          ineq_constraints = {})
      : objective(std::move(obj)),
        equality_constraints(std::move(eq_constraints)),
        inequality_constraints(std::move(ineq_constraints)) {}
};

// Deduction guide for two-argument constructor using an initializer list for
// equality constraints.
template <typename TScalar, DifferentiabilityMode ModeObj,
          DifferentiabilityMode ModeConstr, int TDim>
ConstrainedOptimizationProblem(
    const FunctionExpr<TScalar, ModeObj, TDim> &,
    std::initializer_list<FunctionExpr<TScalar, ModeConstr, TDim>>)
    -> ConstrainedOptimizationProblem<TScalar, ModeObj, ModeConstr, TDim>;

// Deduction guide for three-argument constructor using initializer lists.
template <typename TScalar, DifferentiabilityMode ModeObj,
          DifferentiabilityMode ModeConstr, int TDim>
ConstrainedOptimizationProblem(
    const FunctionExpr<TScalar, ModeObj, TDim> &,
    std::initializer_list<FunctionExpr<TScalar, ModeConstr, TDim>>,
    std::initializer_list<FunctionExpr<TScalar, ModeConstr, TDim>>)
    -> ConstrainedOptimizationProblem<TScalar, ModeObj, ModeConstr, TDim>;

}  // namespace cppoptlib::function

#endif  //  INCLUDE_CPPOPTLIB_FUNCTION_PROBLEM_H_
