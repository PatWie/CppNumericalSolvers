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
#ifndef INCLUDE_CPPOPTLIB_SOLVER_PROGRESS_H_
#define INCLUDE_CPPOPTLIB_SOLVER_PROGRESS_H_
#include <stdint.h>

#include <Eigen/Core>
namespace cppoptlib::solver {
// Status of the solver state.
enum class Status {
  NotStarted = -1,
  Continue = 0,     // Optimization should continue.
  IterationLimit,   // Maximum of allowed iterations has been reached.
  XDeltaViolation,  // Minimum change in parameter vector has been reached.
  FDeltaViolation,  // Minimum change in cost function has been reached.
  GradientNormViolation,  // Minimum norm in gradient vector has been reached.
  HessianConditionViolation,  // Maximum condition number of hessian_t has been
                              // reached.
  Finished                    // Successful finished
};

inline std::ostream &operator<<(std::ostream &stream, const Status &status) {
  switch (status) {
    case Status::NotStarted:
      stream << "Solver not started.";
      break;
    case Status::Continue:
      stream << "Convergence criteria not reached.";
      break;
    case Status::IterationLimit:
      stream << "Iteration limit reached.";
      break;
    case Status::XDeltaViolation:
      stream << "Change in parameter vector too small.";
      break;
    case Status::FDeltaViolation:
      stream << "Change in cost function value too small.";
      break;
    case Status::GradientNormViolation:
      stream << "Gradient vector norm too small.";
      break;
    case Status::HessianConditionViolation:
      stream << "Condition of Hessian/Covariance matrix too large.";
      break;
    case Status::Finished:
      stream << "Finished";
      break;
  }
  return stream;
}

// The progress of the solver.

template <class FunctionType, class StateType>
struct Progress {
  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using MatrixType = typename FunctionType::MatrixType;

  size_t num_iterations = 0;           // Maximum number of allowed iterations.
  ScalarType x_delta = ScalarType{0};  // Minimum change in parameter vector.
  int x_delta_violations = 0;  // Number of violations in pareameter vector.
  ScalarType f_delta = ScalarType{0};  // Minimum change in cost function.
  int f_delta_violations = 0;          // Number of violations in cost function.
  ScalarType gradient_norm = ScalarType{0};  // Minimum norm of gradient vector.
  ScalarType condition_hessian =
      ScalarType{0};  // Maximum condition number of hessian_t.
  ScalarType constraint_threshold =
      ScalarType{0};                   // Minimum norm of constraint violations.
  Status status = Status::NotStarted;  // Status of state.

  Progress() = default;

  // Updates state from function information.
  void Update(const FunctionType &function,
              const StateType &previous_function_state,
              const StateType &current_function_state,
              const Progress<FunctionType, StateType> &stop_progress) {
    const VectorType &current_x = current_function_state.x;
    const VectorType &previous_x = previous_function_state.x;
    VectorType previous_gradient, current_gradient;
    ScalarType previous_value;
    ScalarType current_value;
    if constexpr (FunctionType::Differentiability >=
                  cppoptlib::function::DifferentiabilityMode::First) {
      if constexpr (StateType::NumConstraints > 0) {
        const auto previous_function =
            cppoptlib::function::ToAugmentedLagrangian(
                function, previous_function_state.multiplier_state,
                previous_function_state.penalty_state);
        const auto current_function =
            cppoptlib::function::ToAugmentedLagrangian(
                function, current_function_state.multiplier_state,
                current_function_state.penalty_state);

        previous_value = previous_function(previous_x, &previous_gradient);
        current_value = current_function(current_x, &current_gradient);
      } else {
        previous_value = function(previous_x, &previous_gradient);
        current_value = function(current_x, &current_gradient);
      }
    } else {
      previous_value = function(previous_x);
      current_value = function(current_x);
    }

    num_iterations++;
    f_delta = std::abs(current_value - previous_value);
    x_delta = (current_x - previous_x).template lpNorm<Eigen::Infinity>();

    // Compute gradient norm if the function supports differentiation
    if constexpr (FunctionType::Differentiability >=
                  cppoptlib::function::DifferentiabilityMode::First) {
      gradient_norm = current_gradient.template lpNorm<Eigen::Infinity>();
    }
    // Compute Hessian condition number if the function supports second-order
    // derivatives
    if constexpr (FunctionType::Differentiability ==
                  cppoptlib::function::DifferentiabilityMode::Second) {
      MatrixType current_hessian;
      function(current_x, nullptr, &current_hessian);
      condition_hessian =
          current_hessian.norm() * current_hessian.inverse().norm();
    }

    if ((stop_progress.num_iterations > 0) &&
        (num_iterations > stop_progress.num_iterations)) {
      status = Status::IterationLimit;
      return;
    }
    if constexpr (StateType::NumConstraints > 0) {
      if (std::abs(current_function_state.max_violation) >
          stop_progress.constraint_threshold) {
        status = Status::Continue;
        return;
      }
      status = Status::Finished;  // All constraints satisfied
      return;
    }
    if ((stop_progress.x_delta > 0) && (x_delta < stop_progress.x_delta)) {
      x_delta_violations++;
      if (x_delta_violations >= stop_progress.x_delta_violations) {
        status = Status::XDeltaViolation;
        return;
      }
    } else {
      x_delta_violations = 0;
    }
    if ((stop_progress.f_delta > 0) && (f_delta < stop_progress.f_delta)) {
      f_delta_violations++;
      if (f_delta_violations >= stop_progress.f_delta_violations) {
        status = Status::FDeltaViolation;
        return;
      }
    } else {
      f_delta_violations = 0;
    }
    if constexpr (FunctionType::Differentiability >=
                  cppoptlib::function::DifferentiabilityMode::First) {
      if ((stop_progress.gradient_norm > 0) &&
          (gradient_norm < stop_progress.gradient_norm)) {
        status = Status::GradientNormViolation;
        return;
      }
    }
    if constexpr (FunctionType::Differentiability ==
                  function::DifferentiabilityMode::Second) {
      if ((stop_progress.condition_hessian > 0) &&
          (condition_hessian > stop_progress.condition_hessian)) {
        status = Status::HessianConditionViolation;
        return;
      }
    }
    status = Status::Continue;
  }
};
// Returns the default stopping solver state.
template <class FunctionType, class StateType>
Progress<FunctionType, StateType> DefaultStoppingSolverProgress() {
  Progress<FunctionType, StateType> progress;
  using ScalarType = typename Progress<FunctionType, StateType>::ScalarType;
  progress.num_iterations = 10000;
  progress.x_delta = ScalarType{1e-9};
  progress.x_delta_violations = 5;
  progress.f_delta = ScalarType{1e-9};
  progress.f_delta_violations = 5;
  progress.gradient_norm = ScalarType{1e-6};
  progress.condition_hessian = ScalarType{0};
  progress.constraint_threshold = ScalarType{1e-5};
  progress.status = Status::NotStarted;
  return progress;
}
}  // namespace cppoptlib::solver
#endif  //  INCLUDE_CPPOPTLIB_SOLVER_PROGRESS_H_
