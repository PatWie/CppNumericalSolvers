// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#ifndef INCLUDE_CPPOPTLIB_SOLVER_ISOLVER_H_
#define INCLUDE_CPPOPTLIB_SOLVER_ISOLVER_H_

#include <functional>
#include <iostream>

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
  HessianConditionViolation  // Maximum condition number of Hessian has been
                             // reached.
};

// The state of the solver.
template <class T>
struct State {
  size_t num_iterations = 0;           // Maximum number of allowed iterations.
  T x_delta = T{0};                    // Minimum change in parameter vector.
  T f_delta = T{0};                    // Minimum change in cost function.
  T gradient_norm = T{0};              // Minimum norm of gradient vector.
  T condition_hessian = T{0};          // Maximum condition number of Hessian.
  Status status = Status::NotStarted;  // Status of state.

  State() = default;

  // Resets the state.
  void Reset() {
    num_iterations = 0;
    x_delta = T{0};
    f_delta = T{0};
    gradient_norm = T{0};
    condition_hessian = T{0};
    status = Status::NotStarted;
  }

  // Updates state from function information.
  template <class Scalar, class Vector>
  void UpdateState(const function::State<Scalar, Vector> previous,
                   const function::State<Scalar, Vector> current) {
    f_delta = fabs(current.value - previous.value);
    x_delta = (current.x - previous.x).template lpNorm<Eigen::Infinity>();
    gradient_norm = current.gradient.template lpNorm<Eigen::Infinity>();
  }

  // updates status given another state.
  void UpdateStatus(const State &stop_state) {
    if (num_iterations > stop_state.num_iterations) {
      status = Status::IterationLimit;
      return;
    }
    if (x_delta > stop_state.x_delta) {
      status = Status::XDeltaViolation;
      return;
    }
    if (f_delta > stop_state.f_delta) {
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
  state.x_delta = T{0};
  state.f_delta = T{0};
  state.gradient_norm = T{1e-4};
  state.condition_hessian = T{0};
  state.status = Status::NotStarted;
  return state;
}

template <class Scalar, class Function, class Vector>
auto GetDefaultStepCallback() {
  return [](const State<Scalar> &solver_state,
            const function::State<Scalar, Vector> &function_state,
            const Vector &x) {
    std::cout << "Function" << std::endl;
    std::cout << "  value    " << function_state.value << std::endl;
    std::cout << "Solver" << std::endl;
    std::cout << "  iterations " << solver_state.num_iterations << std::endl;
    std::cout << "  x_delta " << solver_state.x_delta << std::endl;
    std::cout << "  f_delta " << solver_state.f_delta << std::endl;
    std::cout << "  gradient_norm " << solver_state.gradient_norm << std::endl;
    std::cout << "  condition_hessian " << solver_state.condition_hessian
              << std::endl;
  };
}

template <class Scalar, class Function, class Vector>
auto GetEmptyStepCallback() {
  return [](const State<Scalar> &state, const function::State<Scalar, Vector>,
            const Vector &x) {};
}

// Specifies a solver implementation (of a given order) for a given function
template <typename Function, int Ord>
class Solver {
  static_assert(Ord < 3, "");
  static_assert(Ord > 0, "");

 public:
  using Scalar = typename Function::Scalar;
  using Vector = typename Function::Vector;
  using Hessian = typename Function::Hessian;

  using Callback = std::function<void(const State<Scalar> &,
                                      const function::State<Scalar, Vector>,
                                      const Vector &)>;

  explicit Solver(const State<Scalar> &stopping_state =
                      DefaultStoppingSolverState<Scalar>())
      : stopping_state_(stopping_state),
        current_state_(State<Scalar>()),
        step_callback_(GetDefaultStepCallback<Scalar, Function, Vector>()) {}
  virtual ~Solver() = default;

  void SetStepCallback(Callback step_callback) {
    step_callback_ = step_callback;
  }

  virtual void minimize(const Function &function, Vector *x0) {
    function::State<Scalar, Vector> previous;
    function::State<Scalar, Vector> current;

    previous.Update(function, *x0);
    current.Update(function, *x0);

    this->current_state_.Reset();
    do {
      this->step_callback_(this->current_state_, current, *x0);

      this->step(function, x0, previous.gradient);
      current.Update(function, *x0);

      ++this->current_state_.num_iterations;
      this->current_state_.UpdateState(previous, current);
      this->current_state_.UpdateStatus(this->stopping_state_);

      previous = current;
    } while (this->current_state_.status == Status::Continue);
    this->step_callback_(this->current_state_, current, *x0);
  }

  virtual void step(const Function &function, Vector *x0,
                    const Vector &gradient) = 0;

  State<Scalar> CurrentState() const { return this->current_state_; }

 protected:
  State<Scalar> stopping_state_;
  State<Scalar> current_state_;

  Callback step_callback_;
};

};  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_SOLVER_ISOLVER_H_
