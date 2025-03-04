// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#ifndef INCLUDE_CPPOPTLIB_CONSTRAINED_FUNCTION_H_
#define INCLUDE_CPPOPTLIB_CONSTRAINED_FUNCTION_H_

#include <Eigen/Core>
#include <array>
#include <functional>
#include <string>
#include <utility>

#include "function.h"
namespace cppoptlib::function {

template <class function_t, std::size_t NumConstraints>
struct ConstrainedFunction;

template <class base_t>
struct ConstrainedState : public State<base_t> {
  static_assert(base_t::NumConstraints > 0, "need a constrained function base");

  // Self alias for convenience.
  using self_t = ConstrainedState<base_t>;
  // Alias for the unconstrained version of the function base.
  using unconstrained_base_t = FunctionBase<typename base_t::scalar_t,
                                            base_t::Dim, base_t::DiffLevel, 0>;

  typename base_t::vector_t x;
  std::array<typename base_t::scalar_t, base_t::NumConstraints>
      lagrange_multipliers;
  std::array<typename base_t::scalar_t, base_t::NumConstraints> violations;
  typename base_t::scalar_t penalty = 0;

  ConstrainedState() {
    lagrange_multipliers.fill(typename base_t::scalar_t(0));
    violations.fill(typename base_t::scalar_t(0));
    penalty = typename base_t::scalar_t(10);
  }

  ConstrainedState(const self_t &rhs) { CopyState(rhs); }  // nolint

  ConstrainedState operator=(const self_t &rhs) {
    CopyState(rhs);
    return *this;
  }

  void CopyState(const self_t &rhs) {
    x = rhs.x.eval();
    penalty = rhs.penalty;
    lagrange_multipliers = rhs.lagrange_multipliers;
    violations = rhs.violations;
  }

  State<unconstrained_base_t> AsUnconstrained() const {
    State<unconstrained_base_t> state;
    state.x = x.eval();
    return state;
  }
};

template <class cfunction_t>
class UnconstrainedFunctionAdapter
    : public cfunction_t::unconstrained_function_t {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  UnconstrainedFunctionAdapter(cfunction_t constrained_function,
                               typename cfunction_t::state_t constrained_state)
      : constrained_function(constrained_function),
        constrained_state(constrained_state) {}

  typename cfunction_t::unconstrained_function_t::base_t::scalar_t operator()(
      const typename cfunction_t::unconstrained_function_t::base_t::vector_t &x,
      typename cfunction_t::unconstrained_function_t::base_t::vector_t
          *gradient = nullptr

  ) const override {
    return constrained_function(x, constrained_state.lagrange_multipliers,
                                constrained_state.penalty, gradient);
  }

  typename cfunction_t::unconstrained_function_t::state_t GetState(
      const typename cfunction_t::unconstrained_function_t::base_t::vector_t &x)
      const {
    const typename cfunction_t::state_t inner = constrained_function.GetState(
        x, constrained_state.lagrange_multipliers, constrained_state.penalty);
    typename cfunction_t::unconstrained_function_t::state_t unconstrained_state;
    unconstrained_state.x = inner.x;
    return unconstrained_state;
  }

 private:
  cfunction_t constrained_function;
  typename cfunction_t::state_t constrained_state;
};

template <class function_t, std::size_t TNumConstraints>
struct ConstrainedFunction {
  static constexpr int NumConstraints = TNumConstraints;

  using scalar_t = typename function_t::base_t::scalar_t;
  using vector_t = typename function_t::base_t::vector_t;
  using matrix_t = typename function_t::base_t::matrix_t;
  using base_t = FunctionBase<scalar_t, function_t::Dim, function_t::DiffLevel,
                              TNumConstraints>;
  using unconstrained_function_t = function_t;

 public:
  using state_t = ConstrainedState<base_t>;
  static constexpr Differentiability DiffLevel = function_t::DiffLevel;
  ConstrainedFunction(
      const function_t *objective,
      const std::array<const function_t *, TNumConstraints> &constraints)
      : objective_(objective), constraints_(constraints) {}

  scalar_t operator()(
      const typename base_t::vector_t &x,
      std::array<scalar_t, TNumConstraints> lagrange_multipliers,
      scalar_t penalty, typename base_t::vector_t *gradient = nullptr) const {
    scalar_t f;
    vector_t grad;
    if (gradient) {
      f = (*objective_)(x, &grad);
    } else {
      f = (*objective_)(x);
    }

    // Augment the objective with constraint penalties.
    for (std::size_t i = 0; i < TNumConstraints; ++i) {
      vector_t scaled_local_grad;
      scalar_t cost;
      if (gradient) {
        cost = constraints_[i]->operator()(x, &scaled_local_grad);
      } else {
        cost = constraints_[i]->operator()(x);
      }
      const scalar_t violation = cost;
      const scalar_t lambda = lagrange_multipliers[i];

      // Compute the augmented cost for this constraint.
      const scalar_t aug_cost =
          violation + lambda * violation +
          static_cast<scalar_t>(0.5) * penalty * violation * violation;
      f += aug_cost;

      if (gradient) {
        // Augment the gradient only if the constraint is active (i.e. cost >
        // 0).
        const scalar_t a = scalar_t(1) + lambda + penalty * violation;
        scaled_local_grad = a * scaled_local_grad;
        const typename base_t::vector_t aug_grad =
            (cost > scalar_t(0)) ? scaled_local_grad
                                 : base_t::vector_t::Zero(x.size());
        grad += aug_grad;
      }
    }

    if (gradient) {
      *gradient = grad;
    }
    return f;
  }

  state_t GetState(const typename base_t::vector_t &x,
                   std::array<scalar_t, TNumConstraints> lagrange_multipliers,
                   scalar_t penalty) const {
    const typename function_t::state_t objective_state =
        objective_->GetState(x);

    state_t constrained_state;
    constrained_state.x = objective_state.x;
    constrained_state.penalty = penalty;

    for (std::size_t i = 0; i < TNumConstraints; ++i) {
      const scalar_t violation = constraints_[i]->operator()(x);
      constrained_state.violations[i] = violation;
      constrained_state.lagrange_multipliers[i] = lagrange_multipliers[i];
    }

    return constrained_state;
  }

  const function_t *objective_;
  std::array<const function_t *, TNumConstraints> constraints_;
};

template <typename function_t, typename... Constraints>
auto BuildConstrainedProblem(const function_t *objective,
                             const Constraints *...constraints) {
  constexpr std::size_t N = sizeof...(Constraints);
  return ConstrainedFunction<Function<typename function_t::base_t::scalar_t,
                                      function_t::Dim, function_t::DiffLevel>,
                             N>(objective, {constraints...});
}

template <typename T>
struct SquaredPenalty {
  // Returns the penalty: f(x) = x^2
  inline T operator()(const T &x) const { return x * x; }
  // Returns the derivative: f'(x) = 2*x
  inline T derivative(const T &x) const { return 2 * x; }
};

template <typename function_t,
          typename PenaltyPolicy =
              SquaredPenalty<typename function_t::scalar_t>>
class ZeroConstraint : public function_t {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typename function_t::scalar_t operator()(
      const typename function_t::vector_t &x,
      typename function_t::vector_t *gradient = nullptr) const override {
    const PenaltyPolicy policy;
    if (gradient) {
      typename function_t::vector_t constraint_grad;
      // Evaluate the underlying constraint and its gradient.
      const typename function_t::scalar_t c_val =
          constraint_(x, &constraint_grad);
      const typename function_t::scalar_t penalty_value =
          policy(c_val);  // f(c(x)) = c(x)^2
      // Chain rule: gradient = policy.derivative(c(x)) * c'(x)
      *gradient = policy.derivative(c_val) * constraint_grad;
      return penalty_value;
    } else {
      const typename function_t::scalar_t c_val = constraint_(x);
      return policy(c_val);
    }
  }

 private:
  function_t constraint_;
};

template <typename function_t,
          typename PenaltyPolicy =
              SquaredPenalty<typename function_t::scalar_t>>
class NonNegativeConstraint : public function_t {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typename function_t::scalar_t operator()(
      const typename function_t::vector_t &x,
      typename function_t::vector_t *gradient = nullptr) const override {
    const PenaltyPolicy policy;
    // Tolerance below which the constraint is considered violated.
    const typename function_t::scalar_t tol = 1e-7;
    if (gradient) {
      typename function_t::vector_t constraint_grad;
      const typename function_t::scalar_t c_val =
          constraint_(x, &constraint_grad);
      if (c_val >= tol) {
        gradient->setZero();
        return 0;
      } else {
        const typename function_t::scalar_t violation =
            -c_val;  // violation is positive
        const typename function_t::scalar_t penalty_value = policy(violation);
        // Chain rule: gradient = policy.derivative(violation) * (-c'(x))
        *gradient = policy.derivative(violation) * (-constraint_grad);
        return penalty_value;
      }
    } else {
      const typename function_t::scalar_t c_val = constraint_(x);
      if (c_val >= tol) {
        return 0;
      } else {
        const typename function_t::scalar_t violation = -c_val;
        return policy(violation);
      }
    }
  }

 private:
  function_t constraint_;
};

template <typename function_t,
          typename PenaltyPolicy =
              SquaredPenalty<typename function_t::scalar_t>>
class BoxConstraint : public function_t {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Constructs a BoxConstraint with specified lower and upper bounds.
   *
   * @param lower The lower bound vector.
   * @param upper The upper bound vector.
   */
  BoxConstraint(const typename function_t::vector_t &lower,
                const typename function_t::vector_t &upper)
      : lower_(lower), upper_(upper) {}

  /**
   * @brief Evaluates the box constraint penalty using a penalty policy.
   *
   * For each component of x:
   *  - If x[i] < lower_[i], the violation is defined as (lower_[i] - x[i]).
   *  - If x[i] > upper_[i], the violation is defined as (x[i] - upper_[i]).
   *  - Otherwise, no penalty is added.
   *
   * The penalty and its derivative (used for gradient computation) are computed
   * by the policy.
   *
   * @param x The input vector.
   * @param gradient (Optional) Pointer to store the computed gradient.
   * @return The total penalty value.
   */
  typename function_t::scalar_t operator()(
      const typename function_t::vector_t &x,
      typename function_t::vector_t *gradient = nullptr) const override {
    typename function_t::scalar_t total_penalty = 0;
    if (gradient) {
      *gradient = typename function_t::vector_t::Zero(x.size());
    }
    const PenaltyPolicy policy;
    // Evaluate the penalty for each component.
    for (int i = 0; i < x.size(); ++i) {
      if (x[i] < lower_[i]) {
        typename function_t::scalar_t diff = lower_[i] - x[i];
        total_penalty += policy(diff);
        if (gradient) {
          (*gradient)[i] -= policy.derivative(diff);
        }
      } else if (x[i] > upper_[i]) {
        typename function_t::scalar_t diff = x[i] - upper_[i];
        total_penalty += policy(diff);
        if (gradient) {
          (*gradient)[i] += policy.derivative(diff);
        }
      }
    }
    return total_penalty;
  }

 private:
  typename function_t::vector_t lower_;
  typename function_t::vector_t upper_;
};

}  // namespace cppoptlib::function

#endif  //  INCLUDE_CPPOPTLIB_CONSTRAINED_FUNCTION_H_
