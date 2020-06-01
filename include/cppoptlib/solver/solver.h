// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#ifndef INCLUDE_CPPOPTLIB_SOLVER_ISOLVER_H_
#define INCLUDE_CPPOPTLIB_SOLVER_ISOLVER_H_

#include <functional>
#include <iostream>
#include <tuple>

#include "../function.h"

namespace cppoptlib {
namespace solver {

// Status of the solver state.
enum class Status {
  NotStarted = -1,
  Continue = 0,     // Optimization should continue.
  IterationLimit,   // Maximum of allowed iterations has been reached.
  XDeltaViolation,  // Minimum change in parameter vector has been reached.
  FDeltaViolation,  // Minimum chnage in cost function has been reached.
  GradientNormViolation,  // Minimum norm in gradient vector has been reached.
  HessianConditionViolation  // Maximum condition number of HessianT has been
                             // reached.
};

// The state of the solver.
template <class T>
struct State {
  size_t num_iterations = 0;           // Maximum number of allowed iterations.
  T x_delta = T{0};                    // Minimum change in parameter vector.
  T f_delta = T{0};                    // Minimum change in cost function.
  T gradient_norm = T{0};              // Minimum norm of gradient vector.
  T condition_hessian = T{0};          // Maximum condition number of HessianT.
  Status status = Status::NotStarted;  // Status of state.

  State() = default;

  // Updates state from function information.
  template <class ScalarT, class VectorT, class HessianT, int Order>
  void Update(const function::State<ScalarT, VectorT, HessianT, Order>
                  previous_function_state,
              const function::State<ScalarT, VectorT, HessianT, Order>
                  current_function_state,
              const State &stop_state) {
    num_iterations++;
    f_delta =
        fabs(current_function_state.value - previous_function_state.value);
    x_delta = (current_function_state.x - previous_function_state.x)
                  .template lpNorm<Eigen::Infinity>();
    gradient_norm =
        current_function_state.gradient.template lpNorm<Eigen::Infinity>();

    if (num_iterations > stop_state.num_iterations) {
      status = Status::IterationLimit;
      return;
    }
    if (x_delta < stop_state.x_delta) {
      status = Status::XDeltaViolation;
      return;
    }
    if (f_delta < stop_state.f_delta) {
      status = Status::FDeltaViolation;
      return;
    }
    if (gradient_norm < stop_state.gradient_norm) {
      status = Status::GradientNormViolation;
      return;
    }
    if (condition_hessian > stop_state.condition_hessian) {
      status = Status::HessianConditionViolation;
      return;
    }
    status = Status::Continue;
  }
};

// Returns the default stopping solver state.
template <class T>
State<T> DefaultStoppingSolverState() {
  State<T> state;
  state.num_iterations = 10000;
  state.x_delta = T{1e-5};
  state.f_delta = T{1e-5};
  state.gradient_norm = T{1e-4};
  state.condition_hessian = T{0};
  state.status = Status::NotStarted;
  return state;
}

// Returns the defaul callback function.
template <class ScalarT, class VectorT, class HessianT, int Order>
auto GetDefaultStepCallback() {
  return [](
      const function::State<ScalarT, VectorT, HessianT, Order> &function_state,
      const State<ScalarT> &solver_state) {
    std::cout << "Function-State" << std::endl;
    std::cout << "  value    " << function_state.value << std::endl;
    std::cout << "  x    " << function_state.x.transpose() << std::endl;
    std::cout << "Solver-State" << std::endl;
    std::cout << "  iterations " << solver_state.num_iterations << std::endl;
    std::cout << "  x_delta " << solver_state.x_delta << std::endl;
    std::cout << "  f_delta " << solver_state.f_delta << std::endl;
    std::cout << "  gradient_norm " << solver_state.gradient_norm << std::endl;
    std::cout << "  condition_hessian " << solver_state.condition_hessian
              << std::endl;
  };
}

template <class ScalarT, class VectorT, class HessianT, int Order>
auto GetEmptyStepCallback() {
  return [](const function::State<ScalarT, VectorT, HessianT, Order> &,
            const State<ScalarT> &) {};
}

// Specifies a solver implementation (of a given order) for a given function
template <typename TFunction, int TOrder>
class Solver {
  // The solver can be only
  // TOrder==0, given only the objective value
  // TOrder==1, gradient based.
  // TOrder==2, Hessian+Gradient based.
  static_assert(TOrder < 3, "");
  static_assert(TOrder >= 0, "");
  static_assert(TOrder <= TFunction::Order, "");

 public:
  static const int Order = TOrder;
  static const int Dim = TFunction::Dim;
  using FunctionT = TFunction;
  using ScalarT = typename TFunction::ScalarT;
  using VectorT = typename TFunction::VectorT;
  using HessianT = typename TFunction::HessianT;

  using FunctionStateT = function::State<ScalarT, VectorT, HessianT, TOrder>;
  using SolverStateT = State<ScalarT>;
  using CallbackT =
      std::function<void(const FunctionStateT &, const SolverStateT &)>;

  explicit Solver(const State<ScalarT> &stopping_state =
                      DefaultStoppingSolverState<ScalarT>())
      : stopping_state_(stopping_state),
        step_callback_(
            GetDefaultStepCallback<ScalarT, VectorT, HessianT, TOrder>()) {}

  virtual ~Solver() = default;

  // Sets a Callback function which is triggered after each update step.
  void SetStepCallback(CallbackT step_callback) {
    step_callback_ = step_callback;
  }

  // Minimizes a given function and returns the function state
  virtual std::tuple<FunctionStateT, SolverStateT> minimize(
      const TFunction &function, const VectorT &x0) {
    // Solver state during the optimization.
    SolverStateT solver_state;
    // Function state during the optimization.
    FunctionStateT function_state = function.Eval(x0);

    do {
      // Trigger a user-defined callback.
      this->step_callback_(function_state, solver_state);

      // Find next function state.
      FunctionStateT previous_function_state(function_state);
      function_state =
          this->optimization_step(function, previous_function_state);

      // Update current solver state.
      solver_state.Update(previous_function_state, function_state,
                          stopping_state_);
    } while (solver_state.status == Status::Continue);

    // Final Trigger of a user-defined callback.
    this->step_callback_(function_state, solver_state);

    // TODO(patwie): Give C++17 a try.
    return std::make_tuple(function_state, solver_state);
  }

  virtual FunctionStateT optimization_step(const TFunction &function,
                                           const FunctionStateT &state) = 0;

 protected:
  SolverStateT stopping_state_;  // Specifies when to stop.
  CallbackT step_callback_;      // A user-defined callback function.
};

};  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_SOLVER_ISOLVER_H_
