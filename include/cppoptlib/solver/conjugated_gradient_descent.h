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
#ifndef INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_

#include <utility>

#include "../linesearch/armijo.h"
#include "Eigen/Core"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {
template <typename FunctionType>
class ConjugatedGradientDescent
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(
      FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::First ||
          FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::Second,
      "ConjugatedGradientDescent only supports first- or second-order "
      "differentiable functions");

 private:
  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using ProgressType = typename Superclass::ProgressType;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void InitializeSolver(const FunctionType &function,
                        const StateType &initial_state) override {
    function(initial_state.x, &previous_gradient_);
  }

  StateType OptimizationStep(const FunctionType &function,
                             const StateType &current,
                             const ProgressType &progress) override {
    VectorType current_gradient;
    function(current.x, &current_gradient);
    if (progress.num_iterations == 0) {
      search_direction_ = -current_gradient;
    } else {
      const double beta = current_gradient.dot(current_gradient) /
                          (previous_gradient_.dot(previous_gradient_));
      search_direction_ = -current_gradient + beta * search_direction_;
    }
    previous_gradient_ = current_gradient;

    const ScalarType rate = linesearch::Armijo<FunctionType, 1>::Search(
        current.x, search_direction_, function);

    return StateType(current.x + rate * search_direction_);
  }

 private:
  VectorType previous_gradient_;
  VectorType search_direction_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_
