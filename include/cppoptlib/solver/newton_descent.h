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
#ifndef INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_

#include <iostream>

#include "../linesearch/armijo.h"
#include "Eigen/Dense"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {

template <typename FunctionType>
class NewtonDescent
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(FunctionType::Differentiability ==
                    cppoptlib::function::DifferentiabilityMode::Second,
                "NewtonDescent only supports second-order "
                "differentiable functions");

 public:
  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using ProgressType = typename Superclass::ProgressType;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using MatrixType = typename FunctionType::MatrixType;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void InitializeSolver(const FunctionType& /*function*/,
                        const StateType& /*initial_state*/) override {}

  StateType OptimizationStep(const FunctionType& function,
                             const StateType& current,
                             const ProgressType& /*state*/) override {
    constexpr ScalarType safe_guard = 1e-5;

    MatrixType hessian;
    VectorType gradient;
    const ScalarType value = function(current.x, &gradient, &hessian);
    hessian.diagonal().array() += safe_guard;

    const VectorType delta_x = hessian.lu().solve(-gradient);
    // The `(value, gradient, hessian)` triple at `current.x` was just
    // computed; hand it to the line search instead of letting it
    // re-evaluate the full second-order triple at the start point.
    const ScalarType rate =
        linesearch::Armijo<FunctionType, 2>::SearchWithCachedStart(
            current.x, value, gradient, hessian, delta_x, function);

    return StateType(current.x + rate * delta_x);
  }
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
