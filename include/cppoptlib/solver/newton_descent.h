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

 private:
  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using progress_t = typename Superclass::progress_t;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using MatrixType = typename FunctionType::MatrixType;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void InitializeSolver(const FunctionType & /*function*/,
                        const StateType &initial_state) override {
    dim_ = initial_state.x.rows();
  }

  StateType OptimizationStep(const FunctionType &function,
                             const StateType &current,
                             const progress_t & /*state*/) override {
    constexpr ScalarType safe_guard = 1e-5;

    MatrixType hessian;
    VectorType gradient;
    function(current.x, &gradient, &hessian);
    hessian += safe_guard * MatrixType::Identity(dim_, dim_);

    const VectorType delta_x = hessian.lu().solve(-gradient);
    const ScalarType rate = linesearch::Armijo<FunctionType, 2>::Search(
        current.x, delta_x, function);

    return StateType(current.x + rate * delta_x);
  }

 private:
  int dim_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
