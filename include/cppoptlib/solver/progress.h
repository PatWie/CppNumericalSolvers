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
#include <cmath>
#include <cstdlib>
#include <vector>
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

inline std::ostream& operator<<(std::ostream& stream, const Status& status) {
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
  // When true, `f_delta` is interpreted as `factr * epsmch` in Fortran
  // L-BFGS-B 3.0's `|f_k - f_{k+1}| <= factr * epsmch * max(|f_k|,
  // |f_{k+1}|, 1)` convergence test (i.e. a *relative* tolerance scaled
  // by the current function magnitude).  When false, `f_delta` is an
  // absolute threshold.  Default false; `Lbfgsb` sets it true in its own
  // constructor to match Fortran's convergence test exactly.
  bool f_delta_relative = false;
  ScalarType gradient_norm = ScalarType{0};  // Minimum norm of gradient vector.
  // When true, `gradient_norm` is interpreted as a relative tolerance against
  // the current iterate: the solver stops when
  //     `|g|_inf < gradient_norm * max(1, |x|_inf)`.
  // This matches the convergence tests used by Nocedal's Fortran L-BFGS
  // (`gnorm/xnorm <= eps`) and libLBFGS (`||g|| < epsilon * max(1, ||x||)`)
  // and handles badly-scaled problems where `|x|` is large but the residual
  // is already at its floor (e.g. MGH-10 Meyer converges around
  // |x| ~ 6e3, |g|_inf ~ 6e-2).  When false, `gradient_norm` is an absolute
  // threshold, matching LBFGSpp.
  bool gradient_norm_relative = true;
  ScalarType condition_hessian =
      ScalarType{0};  // Maximum condition number of hessian_t.
  ScalarType constraint_threshold =
      ScalarType{0};                   // Minimum norm of constraint violations.
  Status status = Status::NotStarted;  // Status of state.

  // Past-delta stopping: stop when the function value has not decreased
  // meaningfully over the last `past` iterations.  The test fires when
  //     |f_{k-past} - f_k| / max(1, |f_k|) < past_delta.
  // Set `past = 0` to disable (default for backward compatibility).
  // LBFGS-Lite uses past=3, delta=1e-6 and this is a major contributor
  // to its lower nfev count on well-behaved problems.
  int past = 0;
  ScalarType past_delta = ScalarType{1e-6};

  // Ring buffer for the past-delta stopping test (internal state).
  std::vector<ScalarType> past_f_ring_;
  int past_f_pos_ = 0;

  Progress() = default;

  // Updates state from function information.
  //
  // For plain `FunctionState`, the `(value, gradient)` invariant means we
  // can read the numbers we need directly from the two state arguments
  // instead of calling `function()` again.  The `AugmentedLagrangeState`
  // branch still re-evaluates because the augmented-Lagrangian objective
  // is a *composite* built from `function`, the multipliers, and the
  // penalty, none of which are stored as a cached `(value, gradient)` on
  // the state itself.
  void Update(const FunctionType& function,
              const StateType& previous_function_state,
              const StateType& current_function_state,
              const Progress<FunctionType, StateType>& stop_progress) {
    const VectorType& current_x = current_function_state.x;
    const VectorType& previous_x = previous_function_state.x;
    VectorType previous_gradient, current_gradient;
    ScalarType previous_value;
    ScalarType current_value;
    if constexpr (StateType::IsConstrained) {
      // Augmented-Lagrangian path: value/gradient live in the composite
      // objective, not on the state, so we still evaluate here.
      const auto previous_function = cppoptlib::function::ToAugmentedLagrangian(
          function, previous_function_state.multiplier_state,
          previous_function_state.penalty_state);
      const auto current_function = cppoptlib::function::ToAugmentedLagrangian(
          function, current_function_state.multiplier_state,
          current_function_state.penalty_state);

      previous_value = previous_function(previous_x, &previous_gradient);
      current_value = current_function(current_x, &current_gradient);
    } else if constexpr (FunctionType::Differentiability >=
                         cppoptlib::function::DifferentiabilityMode::First) {
      // FunctionState path: read cached (value, gradient) populated by the
      // solver/line search -- no redundant function() calls.
      previous_value = previous_function_state.value;
      current_value = current_function_state.value;
      previous_gradient = previous_function_state.gradient;
      current_gradient = current_function_state.gradient;
    } else {
      // None-mode FunctionState: only value is tracked.
      previous_value = previous_function_state.value;
      current_value = current_function_state.value;
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
    if constexpr (StateType::IsConstrained) {
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
    if ((stop_progress.f_delta > 0) &&
        (f_delta <
         stop_progress.f_delta *
             (stop_progress.f_delta_relative
                  ? std::max({std::abs(current_value), std::abs(previous_value),
                              ScalarType{1}})
                  : ScalarType{1}))) {
      f_delta_violations++;
      if (f_delta_violations >= stop_progress.f_delta_violations) {
        status = Status::FDeltaViolation;
        return;
      }
    } else {
      f_delta_violations = 0;
    }
    // Past-delta stopping: compare current f against f from `past`
    // iterations ago.  The ring buffer is lazily initialized on first use.
    if (stop_progress.past > 0) {
      const int p = stop_progress.past;
      if (static_cast<int>(past_f_ring_.size()) != p) {
        past_f_ring_.assign(p, current_value);
        past_f_pos_ = 0;
      }
      if (static_cast<int>(num_iterations) > p) {
        const ScalarType past_f = past_f_ring_[past_f_pos_];
        const ScalarType rate =
            std::abs(past_f - current_value) /
            std::max(ScalarType{1}, std::abs(current_value));
        if (rate < stop_progress.past_delta) {
          status = Status::FDeltaViolation;
          return;
        }
      }
      past_f_ring_[past_f_pos_] = current_value;
      past_f_pos_ = (past_f_pos_ + 1) % p;
    }
    if constexpr (FunctionType::Differentiability >=
                  cppoptlib::function::DifferentiabilityMode::First) {
      if (stop_progress.gradient_norm > 0) {
        // Scale the gradient-norm threshold by `max(1, |x|_inf)` when the
        // relative test is enabled (default) so badly-scaled problems do
        // not waste line-search evaluations trying to drive `|g|_inf` below
        // an absolute floor that is unachievable at finite precision.
        const ScalarType scale =
            stop_progress.gradient_norm_relative
                ? std::max<ScalarType>(
                      ScalarType{1},
                      current_x.template lpNorm<Eigen::Infinity>())
                : ScalarType{1};
        if (gradient_norm < stop_progress.gradient_norm * scale) {
          status = Status::GradientNormViolation;
          return;
        }
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

#ifdef CPPOPT_SWEEP
  // Parameter sweep mode: read stopping criteria from environment
  // variables so a single binary can be re-run with different settings
  // without recompilation.  See scripts/sweep_params.py.
  auto env_or = [](const char* name, double fallback) -> double {
    const char* v = std::getenv(name);
    return v ? std::atof(v) : fallback;
  };
  auto env_int_or = [](const char* name, int fallback) -> int {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
  };
  progress.x_delta = static_cast<ScalarType>(env_or("CPPOPT_X_DELTA", 1e-9));
  progress.x_delta_violations = env_int_or("CPPOPT_X_DELTA_VIOL", 1);
  progress.f_delta = ScalarType{0};
  progress.f_delta_violations = 1;
  progress.gradient_norm =
      static_cast<ScalarType>(env_or("CPPOPT_GRAD_NORM", 1e-6));
  progress.condition_hessian = ScalarType{0};
  progress.constraint_threshold = ScalarType{1e-5};
  progress.past = env_int_or("CPPOPT_PAST", 5);
  progress.past_delta =
      static_cast<ScalarType>(env_or("CPPOPT_PAST_DELTA", 1e-10));
#else
  progress.x_delta = ScalarType{1e-9};
  // One consecutive x-delta violation is enough for gradient-based solvers:
  // a step that moves `x` by less than `1e-9` while `|g|` has not met the
  // gradient-norm test is a line-search failure, not a legitimate pause.
  // Continuing past it just burns `maxfev` line-search evaluations per
  // iteration on a stuck iterate (seen on MGH 10 Meyer where the tail used
  // to waste ~140 nfev in 7 repeated failed line searches).  Derivative-
  // free solvers like `NelderMead` override this in their own constructor
  // because their simplex does legitimately produce consecutive small
  // x-deltas during contraction.
  progress.x_delta_violations = 1;
  // Unconstrained L-BFGS / BFGS / Gradient-descent etc: no f-delta stopping
  // by default.  Reference implementations (Nocedal `lbfgs_um`, Okazaki's
  // libLBFGS with `past=0`) use gradient-norm alone.  Ill-conditioned
  // unconstrained problems like MGH-03 Powell badly scaled stall the line
  // search with `|Δf| < 1e-9` on the first step while the gradient is
  // still O(1), so an absolute f-delta test fires as a false positive.
  // `Lbfgsb` re-enables it in its own constructor with
  // `f_delta_relative = true` to match Fortran L-BFGS-B's
  // `factr*epsmch*max(|f_k|,|f_{k+1}|,1)` convergence test.
  progress.f_delta = ScalarType{0};
  progress.f_delta_violations = 1;
  // Gradient-norm convergence: `|g|_inf < gradient_norm * max(1, |x|_inf)`.
  // The `gradient_norm_relative = true` default matches Nocedal's
  // `lbfgs_um` (`gnorm/xnorm <= eps`, eps = 1e-5) and libLBFGS
  // (`||g|| < epsilon * max(1, ||x||)`, epsilon = 1e-5).  We keep a
  // slightly tighter default threshold of 1e-6 -- still in the same
  // convergence class as those libraries but one order of magnitude
  // tighter, which matters for well-scaled problems whose `|x|` is
  // O(1) and where a looser threshold leaves measurable residual error.
  progress.gradient_norm = ScalarType{5e-6};
  progress.condition_hessian = ScalarType{0};
  progress.constraint_threshold = ScalarType{1e-5};
  progress.past = 5;
  progress.past_delta = ScalarType{1e-10};
#endif  // CPPOPT_SWEEP
  progress.status = Status::NotStarted;
  return progress;
}
}  // namespace cppoptlib::solver
#endif  //  INCLUDE_CPPOPTLIB_SOLVER_PROGRESS_H_
