// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_
#define INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_

#include <stdint.h>

#include <functional>
#include <iostream>
#include <tuple>
#include <utility>

#include "../function.h"

namespace cppoptlib::solver {

// Status of the solver state.
enum class Status {
  NotStarted = -1,
  Continue = 0,     // Optimization should continue.
  IterationLimit,   // Maximum of allowed iterations has been reached.
  XDeltaViolation,  // Minimum change in parameter vector has been reached.
  FDeltaViolation,  // Minimum chnage in cost function has been reached.
  GradientNormViolation,  // Minimum norm in gradient vector has been reached.
  HessianConditionViolation  // Maximum condition number of hessian_t has been
                             // reached.
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
  }
  return stream;
}

// The progress of the solver.

template <class function_t>
struct Progress {
  using state_t = typename function_t::state_t;
  using scalar_t = typename function_t::scalar_t;

  size_t num_iterations = 0;       // Maximum number of allowed iterations.
  scalar_t x_delta = scalar_t{0};  // Minimum change in parameter vector.
  int x_delta_violations = 0;      // Number of violations in pareameter vector.
  scalar_t f_delta = scalar_t{0};  // Minimum change in cost function.
  int f_delta_violations = 0;      // Number of violations in cost function.
  scalar_t gradient_norm = scalar_t{0};  // Minimum norm of gradient vector.
  scalar_t condition_hessian =
      scalar_t{0};                     // Maximum condition number of hessian_t.
  Status status = Status::NotStarted;  // Status of state.

  Progress() = default;

  // Updates state from function information.
  void Update(const state_t &previous_function_state,
              const state_t &current_function_state,
              const Progress<function_t> &stop_progress) {
    num_iterations++;
    f_delta =
        fabs(current_function_state.value - previous_function_state.value);
    x_delta = (current_function_state.x - previous_function_state.x)
                  .template lpNorm<Eigen::Infinity>();
    gradient_norm =
        current_function_state.gradient.template lpNorm<Eigen::Infinity>();

    if ((stop_progress.num_iterations > 0) &&
        (num_iterations > stop_progress.num_iterations)) {
      status = Status::IterationLimit;
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
    if ((stop_progress.gradient_norm > 0) &&
        (gradient_norm < stop_progress.gradient_norm)) {
      status = Status::GradientNormViolation;
      return;
    }
    // if (previous_function_state.order == 2) {
    //   if ((stop_state.condition_hessian > 0) &&
    //       (condition_hessian > stop_state.condition_hessian)) {
    //     status = Status::HessianConditionViolation;
    //     return;
    //   }
    // }
    status = Status::Continue;
  }
};

// Returns the default stopping solver state.
template <class TFunc>
Progress<TFunc> DefaultStoppingSolverProgress() {
  Progress<TFunc> state;
  using T = typename Progress<TFunc>::scalar_t;
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
template <class function_t>
auto PrintCallback() {
  return [](const typename function_t::state_t &state,
            const Progress<function_t> &progress) {
    std::cout << "Function-State"
              << "\t";
    std::cout << "  value    " << state.value << "\t";
    std::cout << "  x    " << state.x.transpose() << "\t";
    std::cout << "  gradient    " << state.gradient.transpose() << std::endl;
    std::cout << "Solver-State"
              << "\t";
    std::cout << "  iterations " << progress.num_iterations << "\t";
    std::cout << "  x_delta " << progress.x_delta << "\t";
    std::cout << "  f_delta " << progress.f_delta << "\t";
    std::cout << "  gradient_norm " << progress.gradient_norm << "\t";
    std::cout << "  condition_hessian " << progress.condition_hessian
              << std::endl;
  };
}

template <class function_t>
auto NoOpCallback() {
  return [](const typename function_t::state_t & /*state*/,
            const Progress<function_t> & /*progress*/) {};
}

// Specifies a solver implementation (of a given order) for a given function
template <typename function_t>
class Solver {
 public:
  using progress_t = Progress<function_t>;
  using callback_t = std::function<void(const typename function_t::state_t &,
                                        const progress_t &)>;

 private:
  static constexpr int Dim = function_t::Dim;
  using vector_t = typename function_t::vector_t;

 protected:
  using state_t = typename function_t::state_t;

 public:
  explicit Solver(const Progress<function_t> &stopping_progress =
                      DefaultStoppingSolverProgress<function_t>(),
                  callback_t step_callback = NoOpCallback<function_t>())
      : stopping_progress_(stopping_progress),
        step_callback_(std::move(step_callback)) {}

  virtual ~Solver() = default;

  virtual void InitializeSolver(const state_t & /*initial_state*/) = 0;

  // Minimizes a given function and returns the function state
  virtual std::tuple<state_t, progress_t> Minimize(const function_t &function,
                                                   const vector_t &x0) {
    return this->Minimize(function, state_t(function, x0));
  }

  virtual std::tuple<state_t, progress_t> Minimize(
      const function_t &function, const state_t &initial_state) {
    // Solver state during the optimization.
    progress_t solver_state;
    // Function state during the optimization.
    state_t function_state(initial_state);

    this->InitializeSolver(initial_state);

    do {
      // Trigger a user-defined callback.
      this->step_callback_(function_state, solver_state);

      // Find next function state.
      state_t previous_function_state(function_state);
      function_state = this->OptimizationStep(function, previous_function_state,
                                              solver_state);

      // Update current solver state.
      solver_state.Update(previous_function_state, function_state,
                          stopping_progress_);
    } while (solver_state.status == Status::Continue);

    // Final Trigger of a user-defined callback.
    this->step_callback_(function_state, solver_state);

    return {function_state, solver_state};
  }

  virtual state_t OptimizationStep(const function_t &function,
                                   const state_t &current,
                                   const progress_t &state) = 0;

 protected:
  progress_t stopping_progress_;  // Specifies when to stop.
  callback_t step_callback_;      // A user-defined callback function.
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_SOLVER_H_
