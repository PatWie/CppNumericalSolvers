// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_
#define INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_

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
  FDeltaViolation,  // Minimum change in cost function has been reached.
  GradientNormViolation,  // Minimum norm in gradient vector has been reached.
  HessianConditionViolation  // Maximum condition number of hessian_t has been
                             // reached.
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
  }
  return stream;
}

// The state of the solver.
template <class scalar_t>
struct State {
  size_t num_iterations = 0;       // Maximum number of allowed iterations.
  scalar_t x_delta = scalar_t{0};  // Minimum change in parameter vector.
  int x_delta_violations = 0;      // Number of violations in pareameter vector.
  scalar_t f_delta = scalar_t{0};  // Minimum change in cost function.
  int f_delta_violations = 0;      // Number of violations in cost function.
  scalar_t gradient_norm = scalar_t{0};  // Minimum norm of gradient vector.
  scalar_t condition_hessian =
      scalar_t{0};                     // Maximum condition number of hessian_t.
  Status status = Status::NotStarted;  // Status of state.

  State() = default;

  // Updates state from function information.
  template <class vector_t, class hessian_t>
  void Update(const function::State<scalar_t, vector_t, hessian_t>
                  previous_function_state,
              const function::State<scalar_t, vector_t, hessian_t>
                  current_function_state,
              const State& stop_state) {
    num_iterations++;
    f_delta =
        fabs(current_function_state.value - previous_function_state.value);
    x_delta = (current_function_state.x - previous_function_state.x)
                  .template lpNorm<Eigen::Infinity>();
    gradient_norm =
        current_function_state.gradient.template lpNorm<Eigen::Infinity>();

    if ((stop_state.num_iterations > 0) &&
        (num_iterations > stop_state.num_iterations)) {
      status = Status::IterationLimit;
      return;
    }
    if ((stop_state.x_delta > 0) && (x_delta < stop_state.x_delta)) {
      x_delta_violations++;
      if (x_delta_violations >= stop_state.x_delta_violations) {
        status = Status::XDeltaViolation;
        return;
      }
    } else {
      x_delta_violations = 0;
    }
    if ((stop_state.f_delta > 0) && (f_delta < stop_state.f_delta)) {
      f_delta_violations++;
      if (f_delta_violations >= stop_state.f_delta_violations) {
        status = Status::FDeltaViolation;
        return;
      } else {
        f_delta_violations = 0;
      }
    }
    if ((stop_state.gradient_norm > 0) &&
        (gradient_norm < stop_state.gradient_norm)) {
      status = Status::GradientNormViolation;
      return;
    }
    if (previous_function_state.order == 2) {
      if ((stop_state.condition_hessian > 0) &&
          (condition_hessian > stop_state.condition_hessian)) {
        status = Status::HessianConditionViolation;
        return;
      }
    }
    status = Status::Continue;
  }
};

// Returns the default stopping solver state.
template <class T>
State<T> DefaultStoppingSolverState() {
  State<T> state;
  state.num_iterations = 10000;
  state.x_delta = T{1e-9};
  state.x_delta_violations = 5;
  state.f_delta = T{1e-9};
  state.f_delta_violations = 5;
  state.gradient_norm = T{1e-4};
  state.condition_hessian = T{0};
  state.status = Status::NotStarted;
  return state;
}

// Returns the defaul callback function.
template <class scalar_t, class vector_t, class hessian_t>
auto GetDefaultStepCallback() {
  return
      [](const function::State<scalar_t, vector_t, hessian_t>& function_state,
         const State<scalar_t>& solver_state) {
        std::cout << "Function-State"
                  << "\t";
        std::cout << "  value    " << function_state.value << "\t";
        std::cout << "  x    " << function_state.x.transpose() << "\t";
        std::cout << "  gradient    " << function_state.gradient.transpose()
                  << std::endl;
        std::cout << "Solver-State"
                  << "\t";
        std::cout << "  iterations " << solver_state.num_iterations << "\t";
        std::cout << "  x_delta " << solver_state.x_delta << "\t";
        std::cout << "  f_delta " << solver_state.f_delta << "\t";
        std::cout << "  gradient_norm " << solver_state.gradient_norm << "\t";
        std::cout << "  condition_hessian " << solver_state.condition_hessian
                  << std::endl;
      };
}

template <class scalar_t, class vector_t, class hessian_t>
auto GetEmptyStepCallback() {
  return [](const function::State<scalar_t, vector_t, hessian_t>&,
            const State<scalar_t>&) {};
}

// Specifies a solver implementation (of a given order) for a given function
template <typename function_t>
class Solver {
 public:
  using state_t = State<typename function_t::scalar_t>;
  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  using matrix_t = typename function_t::matrix_t;
  using hessian_t = typename function_t::hessian_t;

 private:
  static const int Dim = function_t::Dim;

  using function_state_t = typename function_t::state_t;
  using callback_t =
      std::function<void(const function_state_t&, const state_t&)>;

 public:
  explicit Solver(const State<scalar_t>& stopping_state =
                      DefaultStoppingSolverState<scalar_t>())
      : stopping_state_(stopping_state),
        step_callback_(
            GetDefaultStepCallback<scalar_t, vector_t, hessian_t>()) {}

  virtual ~Solver() = default;

  virtual int Order() const { return 1; }

  // Sets a Callback function which is triggered after each update step.
  void SetStepCallback(callback_t step_callback) {
    step_callback_ = step_callback;
  }

  virtual void InitializeSolver(const function_state_t& /*initial_state*/) {}

  // Minimizes a given function and returns the function state
  virtual std::tuple<function_state_t, state_t> Minimize(
      const function_t& function, const vector_t& x0) {
    return this->Minimize(function, function.Eval(x0, this->Order()));
  }

  virtual std::tuple<function_state_t, state_t> Minimize(
      const function_t& function, const function_state_t& initial_state) {
    // Solver state during the optimization.
    state_t solver_state;
    // Function state during the optimization.
    function_state_t function_state(initial_state);

    this->InitializeSolver(initial_state);

    do {
      // Trigger a user-defined callback.
      this->step_callback_(function_state, solver_state);

      // Find next function state.
      function_state_t previous_function_state(function_state);
      function_state = this->OptimizationStep(function, previous_function_state,
                                              solver_state);

      // Update current solver state.
      solver_state.Update(previous_function_state, function_state,
                          stopping_state_);
    } while (solver_state.status == Status::Continue);

    // Final Trigger of a user-defined callback.
    this->step_callback_(function_state, solver_state);

    return {function_state, solver_state};
  }

  virtual function_state_t OptimizationStep(const function_t& function,
                                            const function_state_t& current,
                                            const state_t& state) = 0;

 protected:
  state_t stopping_state_;    // Specifies when to stop.
  callback_t step_callback_;  // A user-defined callback function.
};

};  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_
