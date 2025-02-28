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

template <class cfunction_t>
struct ConstrainedState : public State<cfunction_t> {
  static_assert(cfunction_t::NumConstraints > -1,
                "need a constrained function");

  using state_t = ConstrainedState<cfunction_t>;

  typename cfunction_t::scalar_t value = 0;
  typename cfunction_t::vector_t x;
  typename cfunction_t::vector_t gradient;

  std::array<typename cfunction_t::scalar_t, cfunction_t::NumConstraints>
      lagrange_multipliers;
  std::array<typename cfunction_t::scalar_t, cfunction_t::NumConstraints>
      violations;
  typename cfunction_t::scalar_t penalty = 0;

  ConstrainedState() {
    lagrange_multipliers.fill(typename cfunction_t::scalar_t(0));
    violations.fill(typename cfunction_t::scalar_t(0));
    penalty = typename cfunction_t::scalar_t(10);
  }

  ConstrainedState(const state_t &rhs) { CopyState(rhs); }  // nolint

  ConstrainedState operator=(const state_t &rhs) {
    CopyState(rhs);
    return *this;
  }

  void CopyState(const state_t &rhs) {
    value = rhs.value;
    x = rhs.x.eval();
    gradient = rhs.gradient.eval();
    penalty = rhs.penalty;
    lagrange_multipliers = rhs.lagrange_multipliers;
    violations = rhs.violations;
  }

  State<typename cfunction_t::unconstrained_function_t> AsUnconstrained()
      const {
    State<typename cfunction_t::unconstrained_function_t> state;
    state.value = value;
    state.x = x.eval();
    state.gradient = gradient.eval();
    return state;
  }
};

template <class cfunction_t>
class UnconstrainedFunctionAdapter
    : public cfunction_t::unconstrained_function_t {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  UnconstrainedFunctionAdapter(cfunction_t original,
                               typename cfunction_t::state_t state)
      : original(original), state(state) {}

  typename cfunction_t::unconstrained_function_t::state_t operator()(
      const typename cfunction_t::types::vector_t &x) const override {
    const typename cfunction_t::state_t inner =
        original(x, state.lagrange_multipliers, state.penalty);
    typename cfunction_t::unconstrained_function_t::state_t outer;
    outer.value = inner.value;
    outer.x = inner.x;
    outer.gradient = inner.gradient;
    return outer;
  }

 private:
  cfunction_t original;
  typename cfunction_t::state_t state;
};

template <class function_t, std::size_t TNumConstraints>
struct ConstrainedFunction {
  static constexpr int NumConstraints = TNumConstraints;

  using types = typename function_t::types;
  using scalar_t = typename function_t::types::scalar_t;
  using vector_t = typename function_t::types::vector_t;
  using matrix_t = typename function_t::types::matrix_t;
  using unconstrained_function_t = function_t;

 public:
  using state_t =
      ConstrainedState<ConstrainedFunction<function_t, NumConstraints>>;
  static constexpr Differentiability diff_level = function_t::diff_level;
  ConstrainedFunction(
      const function_t *objective,
      const std::array<const function_t *, TNumConstraints> &constraints)
      : objective_(objective), constraints_(constraints) {}
  state_t operator()(const typename types::vector_t &x,
                     std::array<scalar_t, TNumConstraints> lagrange_multipliers,
                     scalar_t penalty) const {
    const typename function_t::state_t objective_state =
        objective_->operator()(x);

    state_t constrained_state;
    constrained_state.x = objective_state.x;
    constrained_state.value = objective_state.value;
    constrained_state.gradient = objective_state.gradient;

    // Sum augmented penalties for hard constraints.
    for (std::size_t i = 0; i < TNumConstraints; ++i) {
      const typename function_t::state_t constraint_state =
          constraints_[i]->operator()(x);
      const scalar_t cost = constraint_state.value;
      const scalar_t violation = cost;

      const scalar_t lambda = lagrange_multipliers[i];
      const scalar_t aug_cost =
          violation + lambda * violation +
          static_cast<scalar_t>(0.5) * penalty * violation * violation;
      constrained_state.value += aug_cost;
      // Augmented gradient (only active if the constraint is violated).
      const scalar_t a = scalar_t(1) + lambda + penalty * violation;
      const typename types::vector_t scaled_local_grad =
          a * constraint_state.gradient;
      typename types::vector_t aug_grad = (cost > scalar_t(0))
                                              ? scaled_local_grad
                                              : types::vector_t::Zero(x.size());
      constrained_state.gradient += aug_grad;
      constrained_state.violations[i] = violation;
    }

    return constrained_state;
  }

  const function_t *objective_;
  std::array<const function_t *, TNumConstraints> constraints_;
};

template <typename function_t>
class ZeroConstraint : public function_t {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Construct with the original constraint function.
  explicit ZeroConstraint() {}

  typename function_t::state_t operator()(
      const typename function_t::vector_t &x) const override {
    // Evaluate the original constraint: c(x)
    typename function_t::state_t state = constraint_(x);
    // Save the original constraint value.
    typename function_t::scalar_t c_val = state.value;
    // Transform to squared penalty: f(x) = c(x)^2, with gradient 2*c(x)*c'(x).
    state.value = c_val * c_val;
    state.gradient = 2 * c_val * state.gradient;
    return state;
  }

 private:
  function_t constraint_;
};

template <typename function_t>
class NonNegativeConstraint : public function_t {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Construct with the original constraint function.
  explicit NonNegativeConstraint() {}

  typename function_t::state_t operator()(
      const typename function_t::vector_t &x) const override {
    // Evaluate the original constraint: c(x)
    typename function_t::state_t state = constraint_(x);
    // For inequality constraints, we only penalize when c(x) < 0.
    if (state.value >= 1e-7) {
      // Constraint satisfied; no penalty.
      state.value = 0;
      state.gradient.setZero();
    } else {
      // Constraint violated; penalty = ( - c(x) )^2.
      typename function_t::scalar_t violation = -state.value;
      state.value = violation * violation;
      // Chain rule: gradient = 2*violation * (-c'(x))
      state.gradient = 2 * violation * (-state.gradient);
    }
    return state;
  }

 private:
  function_t constraint_;
};

}  // namespace cppoptlib::function

#endif  //  INCLUDE_CPPOPTLIB_CONSTRAINED_FUNCTION_H_
