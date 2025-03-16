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
#include <iostream>
#include <tuple>
#include <utility>

#include "../function.h"
#include "progress.h"

namespace cppoptlib::solver {

// Returns the defaul callback function.
template <class FunctionType, class StateType>
auto PrintCallback() {
  return [](const FunctionType &function, const StateType &state,
            const Progress<FunctionType, StateType> &progress) {
    std::cout << "Function-State"
              << "\t";
    std::cout << "  value    " << function(state.x) << "\t";
    std::cout << "  x    " << state.x.transpose() << "\t";
    if constexpr (FunctionType::base_t::DifferentiabilityMode >=
                  cppoptlib::function::DifferentiabilityMode::First) {
      typename FunctionType::VectorType gradient;
      function(state.x, &gradient);
      std::cout << "  gradient    " << gradient.transpose() << std::endl;
    }
    std::cout << "Solver-Progress"
              << "\t";
    std::cout << "  iterations " << progress.num_iterations << "\t";
    std::cout << "  x_delta " << progress.x_delta << "\t";
    std::cout << "  f_delta " << progress.f_delta << "\t";
    if constexpr (FunctionType::base_t::DifferentiabilityMode >=
                  cppoptlib::function::DifferentiabilityMode::First) {
      std::cout << "  gradient_norm " << progress.gradient_norm << "\t";
    }
    if constexpr (FunctionType::base_t::DifferentiabilityMode >=
                  cppoptlib::function::DifferentiabilityMode::Second) {
      std::cout << "  condition_hessian " << progress.condition_hessian
                << std::endl;
    }
  };
}

template <class FunctionType, class StateType>
auto NoOpCallback() {
  return [](const FunctionType & /*function*/, const StateType & /*state*/,
            const Progress<FunctionType, StateType> & /*progress*/) {};
}

// Specifies a solver implementation (of a given order) for a given function
template <typename FunctionType, typename StateType>
class Solver {
 public:
  using progress_t = Progress<FunctionType, StateType>;
  //   using callback_t = std::function<void(const FunctionType &func,
  //                                         const typename
  //                                         FunctionType::StateType
  //                                         &, const progress_t &)>;

  // private:
  //   static constexpr int Dim = FunctionType::Dim;
  //   using VectorType = typename FunctionType::VectorType;

  // protected:
  //   using StateType = typename FunctionType::StateType;

 public:
  Solver() = default;
  // explicit Solver(const Progress<FunctionType> &stopping_progress =
  //                     DefaultStoppingSolverProgress<FunctionType>(),
  //                 callback_t step_callback = NoOpCallback<FunctionType>())
  //     : stopping_progress(stopping_progress),
  //       step_callback_(std::move(step_callback)) {}

  virtual ~Solver() = default;

  virtual void InitializeSolver(const FunctionType & /*function*/,
                                const StateType & /*initial_state*/) = 0;

  virtual std::tuple<StateType, Progress<FunctionType, StateType>> Minimize(
      const FunctionType &function, const StateType &function_state) {
    // Solver state during the optimization.
    progress_t solver_state;
    progress_t stopping_progress =
        DefaultStoppingSolverProgress<FunctionType, StateType>();
    // return {function_state, solver_state};
    // // Function state during the optimization.
    StateType current_function_state = function_state;

    // StateType current_function_state(function_state);
    this->InitializeSolver(function, function_state);
    //
    do {
      //   // Trigger a user-defined callback.
      //   this->step_callback_(function, current_function_state, solver_state);
      //
      //   // Find next function state.
      StateType previous_function_state = current_function_state;
      current_function_state = this->OptimizationStep(
          function, previous_function_state, solver_state);
      //
      //   // Update current solver state.
      solver_state.Update(function, previous_function_state,
                          current_function_state, stopping_progress);
    } while (solver_state.status == Status::Continue);
    //
    // // Final Trigger of a user-defined callback.
    // this->step_callback_(function, current_function_state, solver_state);
    //
    // return {current_function_state, solver_state};
    return {current_function_state, solver_state};
  }

  virtual StateType OptimizationStep(const FunctionType &function,
                                     const StateType &current,
                                     const progress_t &state) = 0;

  // progress_t stopping_progress; // Specifies when to stop.
  // protected:
  // callback_t step_callback_; // A user-defined callback function.
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_
