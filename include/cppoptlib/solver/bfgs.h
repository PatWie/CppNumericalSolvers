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
#ifndef INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_
#define INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_

#include <utility>

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {
template <typename FunctionType>
class Bfgs
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(FunctionType::Differentiability ==
                        cppoptlib::function::DifferentiabilityMode::First ||
                    FunctionType::Differentiability ==
                        cppoptlib::function::DifferentiabilityMode::Second,
                "Bfgs only supports first- or second-order "
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
    inverse_hessian_ =
        MatrixType::Identity(initial_state.x.rows(), initial_state.x.rows());
  }

  StateType OptimizationStep(const FunctionType &function,
                             const StateType &current,
                             const progress_t & /*progress*/) override {
    VectorType current_gradient;
    function(current.x, &current_gradient);

    VectorType search_direction = -inverse_hessian_ * current_gradient;

    // If not positive definit re-initialize Hessian.
    const ScalarType phi = current_gradient.dot(search_direction);
    if ((phi > 0) || std::isnan(phi)) {
      // no, we reset the hessian approximation
      inverse_hessian_ = MatrixType::Identity(dim_, dim_);
      search_direction = -current_gradient;
    }

    const ScalarType rate = linesearch::MoreThuente<FunctionType, 1>::Search(
        current.x, search_direction, function);

    const StateType next = StateType(current.x + rate * search_direction);
    VectorType next_gradient;
    function(next.x, &next_gradient);

    // Update inverse Hessian estimate.
    const VectorType s = rate * search_direction;
    const VectorType y = next_gradient - current_gradient;
    const ScalarType rho = 1.0 / y.dot(s);

    inverse_hessian_ =
        inverse_hessian_ -
        rho * (s * (y.transpose() * inverse_hessian_) +
               (inverse_hessian_ * y) * s.transpose()) +
        rho * (rho * y.dot(inverse_hessian_ * y) + 1.0) * (s * s.transpose());

    return next;
  }

 private:
  int dim_;
  MatrixType inverse_hessian_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_
