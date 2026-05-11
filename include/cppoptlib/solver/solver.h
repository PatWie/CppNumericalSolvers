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
#ifndef INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_
#define INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_

#include <stdint.h>

#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../function.h"
#include "progress.h"

namespace cppoptlib::solver {

// Trait: true if `StateType` is a `FunctionState` (and thus participates in
// the `(value, gradient)` invariant) rather than a separate state like
// `AugmentedLagrangeState`.
template <class, class = void>
struct IsFunctionState : std::false_type {};

template <class S>
struct IsFunctionState<S, std::void_t<decltype(std::declval<S>().value),
                                      decltype(std::declval<S>().gradient)>>
    : std::true_type {};

// Returns a callback function that prints progress to the specified stream.
// The output stream (e.g., std::cout, std::cerr, or an std::ofstream)
// is passed as an argument.
template <class FunctionType, class StateType>
auto PrintProgressCallback(std::ostream& output_stream) {
  return [&output_stream](const FunctionType& function, const StateType& state,
                          const Progress<FunctionType, StateType>& progress) {
    const int label_width = 18;  // Width for labels like "Value:", "X Delta:"
    const int num_width = 15;    // Width for numeric values
    const int precision = 6;     // Decimal places for floating-point numbers

    output_stream << std::fixed << std::setprecision(precision);

    // --- Iteration Header ---
    output_stream << "--- Iteration: " << std::setw(5) << std::right
                  << progress.num_iterations << " ---\n";

    // --- Function & State Information ---
    // Read the cached value/gradient from the populated FunctionState.
    // Fall back to one `function()` call only if the state isn't populated
    // (e.g. first iteration with a user-supplied state that didn't go
    // through `Solver::Minimize`'s initial evaluating constructor).
    if constexpr (IsFunctionState<StateType>::value) {
      const auto value = (state.gradient.size() == state.x.size())
                             ? state.value
                             : function(state.x);
      output_stream << std::left << std::setw(label_width)
                    << "  Value:" << std::right << std::setw(num_width) << value
                    << "\n";
    } else {
      output_stream << std::left << std::setw(label_width)
                    << "  Value:" << std::right << std::setw(num_width)
                    << function(state.x) << "\n";
    }

    // Format vector/matrix output using a stringstream to avoid messing up
    // widths easily
    std::stringstream ss_x;
    ss_x << state.x.transpose();

    output_stream << std::left << std::setw(label_width) << "  X:"
                  << " " << ss_x.str() << "\n";

    // --- Gradient (Conditional) ---
    if constexpr (FunctionType::Differentiability >=
                  cppoptlib::function::DifferentiabilityMode::First) {
      typename FunctionType::VectorType gradient;
      if constexpr (IsFunctionState<StateType>::value) {
        if (state.gradient.size() == state.x.size()) {
          gradient = state.gradient;
        } else {
          gradient.resize(state.x.size());
          function(state.x, &gradient);
        }
      } else {
        gradient.resize(state.x.size());
        function(state.x, &gradient);
      }
      std::stringstream ss_grad;
      ss_grad << gradient.transpose();
      output_stream << std::left << std::setw(label_width) << "  Gradient:"
                    << " " << ss_grad.str() << "\n";
      output_stream << std::left << std::setw(label_width)
                    << "  Gradient Norm:" << std::right << std::setw(num_width)
                    << progress.gradient_norm << "\n";
    }

    // --- Solver Progress Information ---
    output_stream << std::left << std::setw(label_width)
                  << "  X Delta:" << std::right << std::setw(num_width)
                  << progress.x_delta << "\n";
    output_stream << std::left << std::setw(label_width)
                  << "  F Delta:" << std::right << std::setw(num_width)
                  << progress.f_delta << "\n";

    // --- Hessian Condition (Conditional) ---
    if constexpr (FunctionType::Differentiability >=
                  cppoptlib::function::DifferentiabilityMode::Second) {
      output_stream << std::left << std::setw(label_width)
                    << "  Hessian Cond.:";
      if (!std::isnan(progress.condition_hessian)) {
        output_stream << std::right << std::setw(num_width)
                      << progress.condition_hessian;
      } else {
        output_stream << std::right << std::setw(num_width) << "N/A";
      }
      output_stream << "\n";
    }

    output_stream << "-------------------------" << std::endl;
  };
}

template <class FunctionType, class StateType>
auto NoOpCallback() {
  return [](const FunctionType& /*function*/, const StateType& /*state*/,
            const Progress<FunctionType, StateType>& /*progress*/) {};
}

// Specifies a solver implementation (of a given order) for a given function
template <typename FunctionTypeT, typename StateTypeT>
class Solver {
 public:
  using StateType = StateTypeT;
  using FunctionType = FunctionTypeT;

  using ProgressType = Progress<FunctionType, StateType>;
  using CallbackType = std::function<void(
      const FunctionType& func, const StateType&, const ProgressType&)>;

  ProgressType stopping_progress;

 public:
  explicit Solver(const ProgressType& progress =
                      DefaultStoppingSolverProgress<FunctionType, StateType>())
      : stopping_progress(progress),
        step_callback_(NoOpCallback<FunctionType, StateType>()) {}

  virtual ~Solver() = default;

  void SetCallback(CallbackType callback) { step_callback_ = callback; }

  virtual void InitializeSolver(const FunctionType& /*function*/,
                                const StateType& /*initial_state*/) = 0;

  virtual std::tuple<StateType, Progress<FunctionType, StateType>> Minimize(
      const FunctionType& function, const StateType& function_state) {
    // Solver state during the optimization.
    ProgressType solver_state;
    // Evaluate `function` once at the user-supplied starting point so the
    // `(value, gradient)` invariant on `FunctionState` is established.  For
    // `AugmentedLagrangeState` and other non-FunctionState types this
    // `if constexpr` branch is skipped and we keep the caller's state as-is.
    StateType current_function_state = function_state;
    if constexpr (IsFunctionState<StateType>::value) {
      current_function_state = StateType(function, function_state.x);
    }

    this->InitializeSolver(function, function_state);

    do {
      this->step_callback_(function, current_function_state, solver_state);

      StateType previous_function_state = current_function_state;
      current_function_state = this->OptimizationStep(
          function, previous_function_state, solver_state);

      // Re-establish the `(value, gradient)` invariant on
      // `current_function_state`.  Solvers that already return a populated
      // state from their line search leave `gradient.size() == x.size()`
      // -- in that case skip the rebuild so the optimization loop pays no
      // redundant evaluation.  Unmigrated solvers return a state with an
      // empty `gradient`; rebuild to keep `Update` and the callback
      // consistent.
      if constexpr (IsFunctionState<StateType>::value) {
        if (current_function_state.gradient.size() !=
            current_function_state.x.size()) {
          current_function_state =
              StateType(function, current_function_state.x);
        }
      }

      solver_state.Update(function, previous_function_state,
                          current_function_state, stopping_progress);
    } while (solver_state.status == Status::Continue);

    this->step_callback_(function, current_function_state, solver_state);
    return {current_function_state, solver_state};
  }

  virtual StateType OptimizationStep(const FunctionType& function,
                                     const StateType& current,
                                     const ProgressType& state) = 0;

  CallbackType step_callback_;  // A user-defined callback function.
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_
