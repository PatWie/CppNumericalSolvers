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
#ifndef INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_
#define INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_

#include <cmath>
#include <limits>
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
                        const StateType& initial_state) override {
    dim_ = initial_state.x.rows();
    inverse_hessian_ =
        MatrixType::Identity(initial_state.x.rows(), initial_state.x.rows());
    fresh_inverse_hessian_ = true;
  }

  StateType OptimizationStep(const FunctionType& function,
                             const StateType& current,
                             const ProgressType& /*progress*/) override {
    constexpr ScalarType eps = std::numeric_limits<ScalarType>::epsilon();

    // Read the cached gradient from the populated FunctionState (no eval).
    const VectorType& current_gradient = current.gradient;

    VectorType search_direction = -inverse_hessian_ * current_gradient;

    // Reset to steepest descent when the approximation is not positive
    // definite (or numerically sick).  Recompute `search_direction` and mark
    // the approximation as fresh so the initial line-search step below is
    // scaled by `1 / ||d||` rather than the quasi-Newton default of `1`.
    const ScalarType phi = current_gradient.dot(search_direction);
    if ((phi > 0) || std::isnan(phi)) {
      inverse_hessian_ = MatrixType::Identity(dim_, dim_);
      search_direction = -current_gradient;
      fresh_inverse_hessian_ = true;
    }

    // Initial line-search step.  On a freshly initialized (identity) inverse
    // Hessian the direction is the raw gradient whose magnitude has no
    // relation to the objective's scale; normalize to `1 / ||d||`.  Once the
    // inverse Hessian carries real curvature information, `alpha_init = 1`
    // is the standard quasi-Newton choice that the Wolfe line search accepts
    // in one evaluation for well-conditioned problems.
    ScalarType alpha_init = ScalarType{1};
    if (fresh_inverse_hessian_) {
      const ScalarType search_direction_norm = search_direction.norm();
      alpha_init = (search_direction_norm > eps)
                       ? ScalarType{1} / search_direction_norm
                       : ScalarType{1};
    }

    // Line search consumes the populated `current` and returns a populated
    // `next` whose `(value, gradient)` are captured from its final internal
    // trial evaluation -- no redundant evaluations at either end.
    const StateType next = linesearch::MoreThuente<FunctionType, 1>::Search(
        current, search_direction, function, alpha_init);

    // Update the inverse Hessian approximation (Nocedal & Wright eqn. 6.17).
    //
    // Guard against degenerate curvature: when `y^T s` is non-positive or
    // effectively zero (which happens if the line search failed and returned
    // `rate = 0`, or if the Wolfe conditions were only marginally met),
    // performing the BFGS update divides by a near-zero `rho` and produces
    // `inf`/`nan` entries that permanently destroy the approximation.  In
    // that case skip the update and keep the previous inverse Hessian.
    const VectorType s = next.x - current.x;
    const VectorType y = next.gradient - current_gradient;
    const ScalarType ys = y.dot(s);
    if (ys > eps * s.norm() * y.norm()) {
      const ScalarType rho = ScalarType{1} / ys;
      const VectorType Hy = inverse_hessian_ * y;
      const ScalarType yHy = y.dot(Hy);
      inverse_hessian_ =
          inverse_hessian_ - rho * (s * Hy.transpose() + Hy * s.transpose()) +
          rho * (rho * yHy + ScalarType{1}) * (s * s.transpose());
      fresh_inverse_hessian_ = false;
    }
    // else: keep the previous inverse_hessian_ -- it was valid last step.

    return next;
  }

 private:
  int dim_ = 0;
  MatrixType inverse_hessian_;
  // `true` after `InitializeSolver` and after any reset to the identity.
  // Used to scale the initial line-search step.
  bool fresh_inverse_hessian_ = true;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_BFGS_H_
