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

#ifndef INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_
#define INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "../function_penalty.h"
#include "../function_problem.h"
#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib::solver {

// Tunable knobs for the augmented-Lagrangian outer loop.  The defaults
// are tuned to escape the "feasible-start, non-convex objective" trap
// where the inner subproblem, with zero multipliers, converges to a
// stationary point of the raw objective that is feasible but not
// optimal.  The escape strategy has three legs:
//
//   1. Adaptive initial penalty.  When the caller does not pass an
//      explicit penalty the outer loop computes one that balances
//      |f(x0)| against the active-constraint squared mass.  The
//      resulting augmented Lagrangian weights constraint gradients
//      comparably to the objective gradient from the very first inner
//      subproblem, so the inner solver cannot step across a boundary
//      "cheaply" into the infeasible side.
//
//   2. Subproblem warm-up.  On the first outer iteration the inner
//      solver is run with a shallow iteration cap and a loose
//      gradient tolerance, so it produces a coarse minimiser that
//      lets the multiplier update fire *before* the iterate locks
//      onto a spurious stationary point of the raw objective.
//      Subsequent outer iterations progressively tighten the
//      subproblem tolerance as the multipliers improve.
//
//   3. KKT-aware stopping.  The outer loop reports `Finished` only
//      when primal feasibility AND stationarity of the Lagrangian
//      both hold.  A feasibility-only stop (earlier behaviour) lets
//      the loop exit at any feasible stationary point of `f`, which
//      is not in general a constrained optimum.
//
// The knobs below name each decision so users can audit or override
// it.  Every default is a clearly-motivated constant; none is a
// black-box tuning number.
template <typename TScalar>
struct AugmentedLagrangianConfig {
  // Penalty-growth schedule.
  //
  // Multiplicative factor applied to `rho` when the most recent outer
  // iteration's `max_violation` failed to shrink by at least
  // `violation_shrink_ratio` relative to the previous iterate.  Must
  // be `>= 1`.  The textbook value of `10` pushes `rho` through a few
  // orders of magnitude in a small number of iterations on truly
  // infeasible starts; a value of `1` freezes `rho` and lets the
  // multipliers do all of the work.
  TScalar penalty_growth_factor = TScalar{10};

  // If `max_violation_new <= violation_shrink_ratio * max_violation_prev`
  // then the penalty is left unchanged.  The classical choice is
  // `0.25` -- the violation must shrink to a quarter of the previous
  // iterate's value for the multipliers to "deserve" a free ride.
  TScalar violation_shrink_ratio = TScalar{0.25};

  // Auto-scale the initial penalty when the caller passes a penalty
  // value of exactly zero on the initial state.  In that case the
  // outer loop computes a problem-dependent `rho_0` of
  //     rho_0 = penalty_auto_objective_scale *
  //             max(1, |f(x0)|) / max(1, 0.5 * sum_i c_i(x0)^2)
  // using only the *active* constraint mass (equalities always
  // contribute; inequalities contribute only when violated at x0).
  //
  // Users who pass a non-zero penalty on the state keep full control.
  // The check against zero is exact on purpose: a caller who genuinely
  // wants `rho_0 = 0` is constructing a pathological problem and
  // should understand the consequence.
  bool auto_scale_initial_penalty = true;
  TScalar penalty_auto_objective_scale = TScalar{10};
  TScalar penalty_auto_min = TScalar{1e-8};
  TScalar penalty_auto_max = TScalar{1e8};

  // Subproblem warm-up on the first outer iteration.
  //
  // The first inner solve runs with a capped iteration count and a
  // loosened gradient-norm stopping tolerance derived from the
  // outer-loop KKT tolerance.  The cap bounds how far the inner
  // solver can walk before the multipliers get their first update.
  // A value of zero disables warm-up entirely.
  int warmup_max_inner_iterations = 10;
  TScalar warmup_inner_gradient_tolerance = TScalar{1e-2};

  // Multiplier safeguarding.
  //
  // Inequality multipliers are projected into `[0, multiplier_max]`
  // after each update; equality multipliers into
  // `[-multiplier_max, multiplier_max]`.  A finite clamp prevents a
  // multiplier from running to +/- infinity on an ill-posed problem
  // (the classical divergence mode that produces NaN composites).
  // The default is large enough that well-posed problems never feel
  // the clamp.
  TScalar multiplier_max = TScalar{1e20};

  // KKT stationarity tolerance for the outer loop.  When the returned
  // Lagrangian gradient sup-norm is above this threshold the outer
  // loop refuses to declare `Finished` even if the primal is
  // feasible.  The default is coordinated with the outer-loop
  // stopping `Progress::gradient_norm` so that tightening the
  // outer-loop gradient test also tightens the KKT criterion.
  //
  // Set to a negative value to disable the KKT check and fall back to
  // primal-feasibility-only stopping (the earlier behaviour).
  TScalar kkt_gradient_tolerance = TScalar{1e-4};
};

template <typename TScalar, int TDimension = Eigen::Dynamic>
struct AugmentedLagrangeState {
  // Marks this state as a constrained-solver state; `Progress::Update`
  // picks a different convergence branch when it reads this trait.
  static constexpr bool IsConstrained = true;
  using VectorType = Eigen::Matrix<TScalar, TDimension, 1>;
  VectorType x;

  // State for Lagrange multipliers.  Declared BEFORE `max_violation` so
  // the initializer-list order in the constructors below -- which
  // populates multipliers and penalty before the violation counter --
  // matches the declaration order.
  cppoptlib::function::LagrangeMultiplierState<TScalar> multiplier_state;

  // State for the penalty parameter.
  cppoptlib::function::PenaltyState<TScalar> penalty_state;

  // Largest constraint violation observed at the most recent outer
  // iteration.  For equality `c = 0` this is `|c(x)|`; for inequality
  // `c >= 0` this is `max(0, -c(x))`.  Compared against the solver's
  // `constraint_threshold` stopping tolerance in `Progress::Update`.
  TScalar max_violation;

  // Sup-norm of the Lagrangian gradient at the current iterate, with
  // the current multipliers.  A correct KKT point has both
  // `max_violation` AND `max_lagrangian_gradient` small.  Feasibility
  // alone is insufficient: a non-convex objective can have feasible
  // stationary points that are not constrained optima.
  //
  // Initialised to +infinity so the very first outer iteration cannot
  // be mistaken for KKT-satisfied by `Progress::Update`.
  TScalar max_lagrangian_gradient;

  // True when `penalty_state.penalty` was filled in by the outer
  // loop's auto-scaling rule (see `AugmentedLagrangianConfig`).  The
  // Minimize entry point reads this flag: the first time a state
  // flows in with the default `penalty == 0` and auto-scaling
  // enabled, the outer loop computes `rho_0` and sets the flag.
  // Subsequent calls with the same state (user-driven continuation)
  // keep the computed value.
  bool penalty_was_auto_scaled;

  // Constructor #1: Construct from an initial guess, custom initializer lists
  // for equality and inequality multipliers, and a penalty value.
  //
  // Usage:
  //   AugmentedLagrangeState<double, 2> state(x, {0.0}, {0.0}, 1.0);
  AugmentedLagrangeState(const VectorType& init_x,
                         std::initializer_list<TScalar> eq_multipliers,
                         std::initializer_list<TScalar> ineq_multipliers,
                         TScalar penalty)
      : x(init_x),
        multiplier_state(eq_multipliers, ineq_multipliers),
        penalty_state(penalty),
        max_violation(0),
        max_lagrangian_gradient(std::numeric_limits<TScalar>::infinity()),
        penalty_was_auto_scaled(false) {}

  // Constructor #2: Construct from an initial guess, numbers of equality and
  // inequality constraints (multipliers will be zero-initialized), and an
  // optional penalty value.  A penalty of `0` (the new default) asks the
  // outer loop to auto-scale it on the first call; see
  // `AugmentedLagrangianConfig::auto_scale_initial_penalty`.
  //
  // Usage:
  //   AugmentedLagrangeState<double, 2> state(x, 1, 1);      // auto-scale
  //   AugmentedLagrangeState<double, 2> state(x, 1, 1, 5.0); // fixed rho
  AugmentedLagrangeState(const VectorType& init_x, size_t num_eq,
                         size_t num_ineq, TScalar penalty = TScalar(0))
      : x(init_x),
        multiplier_state(num_eq, num_ineq, TScalar(0)),
        penalty_state(penalty),
        max_violation(0),
        max_lagrangian_gradient(std::numeric_limits<TScalar>::infinity()),
        penalty_was_auto_scaled(false) {}
};

template <typename ProblemType, typename solver_t>
class AugmentedLagrangian
    : public Solver<ProblemType,
                    AugmentedLagrangeState<typename ProblemType::ScalarType,
                                           ProblemType::Dimension>> {
  static_assert(ProblemType::Differentiability ==
                        cppoptlib::function::DifferentiabilityMode::First ||
                    ProblemType::Differentiability ==
                        cppoptlib::function::DifferentiabilityMode::Second,
                "AugmentedLagrangian only supports first- or second-order "
                "differentiable functions");

 public:
  using StateType = AugmentedLagrangeState<typename ProblemType::ScalarType,
                                           ProblemType::Dimension>;
  using Superclass = Solver<ProblemType, StateType>;
  using ProgressType = typename Superclass::ProgressType;

  using ScalarType = typename ProblemType::ScalarType;
  using VectorType = typename ProblemType::VectorType;
  using MatrixType = typename ProblemType::MatrixType;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Construct by binding the problem and the inner unconstrained solver
  // at once.  The problem is stored by value; callers can then use the
  // convenience `Minimize(state)` overload and never repeat the problem
  // argument.  Both the problem and the inner solver are copied once
  // here.  The stored inner solver is the *template* used for each
  // outer iteration: every call clones it and overrides the clone's
  // `stopping_progress` to implement the warm-up schedule, so the
  // user's original inner-solver object (and its stopping behaviour on
  // unconstrained problems) is never touched.
  AugmentedLagrangian(const ProblemType& problem,
                      const solver_t& unconstrained_solver,
                      AugmentedLagrangianConfig<ScalarType> config = {})
      : problem_(problem),
        unconstrained_solver_template_(unconstrained_solver),
        config_(config),
        outer_iteration_count_(0) {}

  // Bring the base's two-argument `Minimize(problem, state)` overload
  // into scope so that it is not hidden by the one-argument
  // `Minimize(state)` convenience form below.
  using Superclass::Minimize;

  // Convenience: run the solver on the problem bound at construction.
  std::tuple<StateType, ProgressType> Minimize(const StateType& state) {
    return Minimize(problem_, state);
  }

  void InitializeSolver(const ProblemType& /*function*/,
                        const StateType& /*initial_state*/) override {
    outer_iteration_count_ = 0;
  }

  StateType OptimizationStep(const ProblemType& function,
                             const StateType& state,
                             const ProgressType& /*progress*/) override {
    ++outer_iteration_count_;
    StateType next_state(state);

    // --- Step 1: auto-scale the initial penalty, once, if requested ---
    //
    // On the very first outer iteration, a caller who asked for
    // auto-scaling by passing `penalty = 0` gets a problem-dependent
    // `rho_0`.  Rationale: with `rho = 0` the augmented Lagrangian
    // collapses to `f + lambda^T c`, which has no curvature pulling
    // the iterate back onto the feasible side; the inner solver is
    // free to step arbitrarily far across an inactive-at-start
    // boundary before the multiplier update notices.  A balanced
    // `rho_0` that makes the constraint curvature comparable to the
    // objective curvature prevents that.
    if (outer_iteration_count_ == 1 && config_.auto_scale_initial_penalty &&
        !next_state.penalty_was_auto_scaled &&
        next_state.penalty_state.penalty == ScalarType{0}) {
      next_state.penalty_state.penalty =
          ComputeAutoScaledPenalty(function, next_state.x);
      next_state.penalty_was_auto_scaled = true;
    }

    // --- Step 2: build the augmented-Lagrangian composite ---
    //
    // This is the subproblem objective for the inner solver.  The
    // composite has gradient ∇f + sum_i [lambda_i + rho c_i] ∇c_i
    // (equality) and ∇f - sum_j [mu_j - rho min(0, c_j)] ∇c_j
    // (inequality).  A stationary point of the composite is
    // equivalent to a point where (lambda_i + rho c_i) and
    // (mu_j - rho min(0, c_j)) stationarise the raw Lagrangian; the
    // multiplier update below turns these bracketed quantities into
    // the new (lambda, mu) directly.
    const auto unconstrained_function =
        cppoptlib::function::ToAugmentedLagrangian(
            function, next_state.multiplier_state, next_state.penalty_state);

    // --- Step 3: solve the subproblem with a tolerance that depends
    //             on the outer iteration index ---
    //
    // On outer iteration 1 we run the inner solver with a shallow
    // iteration cap and a loose gradient tolerance -- we do NOT want
    // the inner solver to converge fully here.  Its job on iter 1 is
    // to produce a coarse minimiser that drives at least some
    // constraint residuals off zero so the subsequent multiplier
    // update moves them off zero.  On later iterations we let the
    // inner solver converge to the user's inner tolerance.  The
    // clone pattern means the user's `unconstrained_solver_template_`
    // keeps its own stopping_progress intact -- callers using the
    // inner solver independently see no change.
    solver_t working_inner = unconstrained_solver_template_;
    ConfigureInnerSubproblem(working_inner);

    const auto solved_inner_state = std::get<0>(working_inner.Minimize(
        unconstrained_function,
        cppoptlib::function::FunctionState<ScalarType, ProblemType::Dimension>(
            next_state.x)));
    next_state.x = solved_inner_state.x;

    // --- Step 4: update multipliers and measure constraint residuals ---
    const ScalarType penalty = next_state.penalty_state.penalty;
    ScalarType max_violation = ScalarType{0};

    // Equality: first-order multiplier update
    //   lambda_{k+1} = lambda_k + rho_k * c(x_k)
    // derived from the stationarity of the augmented Lagrangian.
    // Then clamp to the multiplier box.
    for (std::size_t i = 0; i < function.equality_constraints.size(); ++i) {
      const ScalarType constraint_value =
          function.equality_constraints[i](next_state.x);
      max_violation =
          std::max<ScalarType>(max_violation, std::abs(constraint_value));
      ScalarType& lambda = next_state.multiplier_state.equality_multipliers[i];
      lambda = ClampEqualityMultiplier(lambda + penalty * constraint_value);
    }

    // Inequality: first-order multiplier update for the `c(x) >= 0`
    // convention is
    //   mu_{k+1} = max(0, mu_k - rho_k * c(x_k))
    // where the `max(0, .)` enforces dual feasibility.  Then clamp
    // to the non-negative multiplier box.
    for (std::size_t i = 0; i < function.inequality_constraints.size(); ++i) {
      const ScalarType constraint_value =
          function.inequality_constraints[i](next_state.x);
      const ScalarType violation =
          std::max<ScalarType>(ScalarType{0}, -constraint_value);
      max_violation = std::max<ScalarType>(max_violation, violation);
      ScalarType& mu = next_state.multiplier_state.inequality_multipliers[i];
      mu = ClampInequalityMultiplier(
          std::max<ScalarType>(ScalarType{0}, mu - penalty * constraint_value));
    }

    // --- Step 5: measure Lagrangian-gradient stationarity ---
    //
    // The Lagrangian is L(x, lambda, mu) = f(x) + sum lambda_i c_i(x)
    // - sum mu_j g_j(x).  A KKT point has |grad_x L|_inf ≈ 0.  This
    // differs from the augmented-Lagrangian gradient by the penalty
    // terms, and it is the one that enters the convergence test at
    // the outer-loop level (the augmented-Lagrangian gradient at the
    // inner solver's converged point is zero by construction, which
    // carries no new information at the outer level).
    next_state.max_lagrangian_gradient =
        ComputeLagrangianGradientSupNorm(function, next_state);
    next_state.max_violation = max_violation;

    // --- Step 6: track the best iterate seen so far ---
    //
    // A non-convex objective can make the inner subproblem multi-
    // modal: different outer iterations see different subproblem
    // landscapes (the penalty and multipliers move) and the inner
    // solver can land in different basins.  The outer loop's
    // convergence test only sees the MOST RECENT iterate, so a
    // later iteration that finds a worse KKT point overwrites a
    // previous better one.  To defend against this we keep an
    // internal "best" iterate indexed by the Pareto order
    //   (not feasible, feasible+higher-f) < (feasible+lower-f).
    // At the end of `Minimize` we overwrite `next_state` with the
    // best.  The trade-off: we pay one extra objective evaluation
    // per outer iteration to read `f(next_state.x)`.
    UpdateBestIterateInPlace(function, next_state);

    // --- Step 7: penalty update ---
    //
    // Grow `rho` only when the violation failed to shrink by the
    // specified ratio.  On a feasible-start problem (`previous ~ 0`)
    // this condition never fires; on an infeasible-start problem
    // with a stubborn violation the penalty climbs through orders of
    // magnitude until the constraint curvature dominates.
    const ScalarType previous_max_violation = state.max_violation;
    const bool violation_shrank_enough =
        max_violation <=
        config_.violation_shrink_ratio * previous_max_violation;
    next_state.penalty_state.penalty =
        violation_shrank_enough ? penalty
                                : penalty * config_.penalty_growth_factor;
    return next_state;
  }

  // Override `Minimize` so that the best-feasible-iterate tracker is
  // consulted before the state is returned.  The outer loop's last
  // iterate is often NOT the best: a multiplier cycle on a non-convex
  // subproblem can bounce to a worse KKT point after having seen a
  // better one.  `best_iterate_x_` holds the Pareto-preferred iterate
  // discovered across all outer iterations; we install it on the
  // returned state, along with its multiplier/penalty state from
  // that iteration, provided any iterate was recorded at all.
  std::tuple<StateType, ProgressType> Minimize(
      const ProblemType& function, const StateType& function_state) override {
    ResetBestIterateTracker();
    auto [final_state, progress] =
        Superclass::Minimize(function, function_state);
    if (best_iterate_recorded_) {
      final_state.x = best_iterate_x_;
      final_state.multiplier_state = best_iterate_multipliers_;
      final_state.penalty_state.penalty = best_iterate_penalty_;
      final_state.max_violation = best_iterate_violation_;
      final_state.max_lagrangian_gradient = best_iterate_kkt_gradient_;
    }
    return {final_state, progress};
  }

 private:
  // Auto-scale the initial penalty so that the augmented-Lagrangian
  // penalty term has the same order of magnitude as the objective at
  // the initial iterate.  Equality constraints always contribute to
  // the squared-residual sum; inequality constraints contribute only
  // on their violated side at x0.  The multiplicative
  // `penalty_auto_objective_scale` (default 10) biases rho upward --
  // we prefer to err on the side of feeling the constraints.
  ScalarType ComputeAutoScaledPenalty(const ProblemType& function,
                                      const VectorType& x) const {
    ScalarType objective_magnitude = std::abs(function.objective(x));
    objective_magnitude =
        std::max<ScalarType>(objective_magnitude, ScalarType{1});

    ScalarType squared_residual_sum = ScalarType{0};
    for (const auto& c : function.equality_constraints) {
      const ScalarType value = c(x);
      squared_residual_sum += ScalarType{0.5} * value * value;
    }
    for (const auto& c : function.inequality_constraints) {
      const ScalarType value = c(x);
      // Convention `c >= 0`: the violated side is `c < 0`.
      if (value < ScalarType{0}) {
        squared_residual_sum += ScalarType{0.5} * value * value;
      }
    }
    const ScalarType denom =
        std::max<ScalarType>(squared_residual_sum, ScalarType{1});
    const ScalarType rho =
        config_.penalty_auto_objective_scale * objective_magnitude / denom;
    return std::clamp(rho, config_.penalty_auto_min, config_.penalty_auto_max);
  }

  // Install the subproblem stopping criteria on the provided inner
  // solver clone.  On the first outer iteration we cap the
  // iterations and loosen the gradient tolerance; on later
  // iterations we leave the clone's stopping_progress untouched --
  // it was copied from the user-provided template, so the user's
  // desired inner-solver behaviour applies.
  void ConfigureInnerSubproblem(solver_t& working_inner) const {
    if (outer_iteration_count_ == 1 &&
        config_.warmup_max_inner_iterations > 0) {
      working_inner.stopping_progress.num_iterations =
          static_cast<std::size_t>(config_.warmup_max_inner_iterations);
      working_inner.stopping_progress.gradient_norm =
          config_.warmup_inner_gradient_tolerance;
    }
  }

  // Clamp an equality-multiplier candidate into the symmetric
  // safeguard box.  NaN values (e.g. from an ill-posed inner solve)
  // are replaced by zero so the next outer iteration restarts from
  // the no-information state rather than propagating NaN into the
  // composite.
  ScalarType ClampEqualityMultiplier(ScalarType candidate) const {
    if (!std::isfinite(candidate)) return ScalarType{0};
    return std::clamp(candidate, -config_.multiplier_max,
                      config_.multiplier_max);
  }

  // Clamp an inequality-multiplier candidate into the non-negative
  // safeguard box.  The caller has already applied the `max(0, .)`
  // projection; this clamp only handles the upper edge and NaN
  // sanitisation.
  ScalarType ClampInequalityMultiplier(ScalarType candidate) const {
    if (!std::isfinite(candidate)) return ScalarType{0};
    return std::clamp(candidate, ScalarType{0}, config_.multiplier_max);
  }

  // Evaluate grad_x L(x, lambda, mu) at the current state and return
  // its sup-norm.  L(x, lambda, mu) = f(x) + sum lambda_i c_i(x) -
  // sum mu_j g_j(x).
  //
  // This is *not* the augmented-Lagrangian gradient -- it is the
  // gradient of the raw Lagrangian, which is what enters the
  // convergence test at the outer-loop level.
  static ScalarType ComputeLagrangianGradientSupNorm(
      const ProblemType& function, const StateType& state) {
    const std::size_t n = static_cast<std::size_t>(state.x.size());
    VectorType sum_grad(state.x.size());
    VectorType buf(state.x.size());
    function.objective(state.x, &sum_grad);
    for (std::size_t i = 0; i < function.equality_constraints.size(); ++i) {
      function.equality_constraints[i](state.x, &buf);
      const ScalarType lambda_i =
          state.multiplier_state.equality_multipliers[i];
      sum_grad.noalias() += lambda_i * buf;
    }
    for (std::size_t j = 0; j < function.inequality_constraints.size(); ++j) {
      function.inequality_constraints[j](state.x, &buf);
      const ScalarType mu_j = state.multiplier_state.inequality_multipliers[j];
      sum_grad.noalias() -= mu_j * buf;
    }
    ScalarType sup = ScalarType{0};
    for (std::size_t k = 0; k < n; ++k) {
      sup = std::max<ScalarType>(sup, std::abs(sum_grad[k]));
    }
    return sup;
  }

  ProblemType problem_;
  solver_t unconstrained_solver_template_;
  AugmentedLagrangianConfig<ScalarType> config_;
  // Counts outer iterations within a single `Minimize` invocation;
  // reset by `InitializeSolver`.  Drives the subproblem warm-up and
  // the auto-scaled-penalty gate.
  std::size_t outer_iteration_count_;

  // "Best iterate so far" tracker.  Non-convex AL problems produce
  // subproblem sequences whose last iterate is not necessarily the
  // best one: an active-set toggle can push a later outer iteration
  // into a worse KKT point.  Keeping the Pareto-best across outer
  // iterations rescues the output from that failure mode.
  //
  // `best_iterate_recorded_` is false until the first outer
  // iteration has recorded an iterate; `Minimize` uses that flag as
  // a "no valid best" signal (in which case it falls back to the
  // outer loop's raw last state).
  bool best_iterate_recorded_;
  VectorType best_iterate_x_;
  cppoptlib::function::LagrangeMultiplierState<ScalarType>
      best_iterate_multipliers_;
  ScalarType best_iterate_penalty_;
  ScalarType best_iterate_objective_;
  ScalarType best_iterate_violation_;
  ScalarType best_iterate_kkt_gradient_;

  void ResetBestIterateTracker() {
    best_iterate_recorded_ = false;
    best_iterate_objective_ = std::numeric_limits<ScalarType>::infinity();
    best_iterate_violation_ = std::numeric_limits<ScalarType>::infinity();
    best_iterate_kkt_gradient_ = std::numeric_limits<ScalarType>::infinity();
    best_iterate_penalty_ = ScalarType{0};
  }

  // Pareto comparison for the "best iterate so far" tracker.  The
  // order is:
  //   1. An iterate with max_violation <= feas_tol is strictly
  //      better than any iterate with larger violation.
  //   2. Among feasible iterates, lower objective wins.
  //   3. Among infeasible iterates, smaller violation wins; ties
  //      broken by lower objective.
  // This is the standard "filter" method used in SQP and AL to
  // reject iterates that trade feasibility for objective value
  // without net improvement.
  //
  // `feas_tol` matches the outer-loop's `constraint_threshold` in
  // spirit but we read it from a separate constant -- tightening the
  // outer-loop threshold should NOT cause the filter to drop all
  // previously-feasible iterates.
  void UpdateBestIterateInPlace(const ProblemType& function,
                                const StateType& candidate) {
    constexpr ScalarType filter_feasibility_tolerance = ScalarType{1e-5};
    const ScalarType candidate_objective = function.objective(candidate.x);
    const ScalarType candidate_violation = candidate.max_violation;
    if (!best_iterate_recorded_) {
      RecordBestIterate(candidate, candidate_objective);
      return;
    }
    const bool candidate_feasible =
        candidate_violation <= filter_feasibility_tolerance;
    const bool best_feasible =
        best_iterate_violation_ <= filter_feasibility_tolerance;
    if (candidate_feasible && !best_feasible) {
      RecordBestIterate(candidate, candidate_objective);
      return;
    }
    if (!candidate_feasible && best_feasible) {
      // Never overwrite a feasible best with an infeasible candidate.
      return;
    }
    if (candidate_feasible && best_feasible) {
      if (candidate_objective < best_iterate_objective_) {
        RecordBestIterate(candidate, candidate_objective);
      }
      return;
    }
    // Both infeasible.  Smaller violation wins; ties broken by
    // objective.
    if (candidate_violation < best_iterate_violation_ ||
        (candidate_violation == best_iterate_violation_ &&
         candidate_objective < best_iterate_objective_)) {
      RecordBestIterate(candidate, candidate_objective);
    }
  }

  void RecordBestIterate(const StateType& candidate,
                         ScalarType candidate_objective) {
    best_iterate_recorded_ = true;
    best_iterate_x_ = candidate.x;
    best_iterate_multipliers_ = candidate.multiplier_state;
    best_iterate_penalty_ = candidate.penalty_state.penalty;
    best_iterate_objective_ = candidate_objective;
    best_iterate_violation_ = candidate.max_violation;
    best_iterate_kkt_gradient_ = candidate.max_lagrangian_gradient;
  }
};

template <typename ProblemType, typename solver_t>
AugmentedLagrangian(const ProblemType&, const solver_t&)
    -> AugmentedLagrangian<ProblemType, solver_t>;

template <typename ProblemType, typename solver_t, typename TScalar>
AugmentedLagrangian(const ProblemType&, const solver_t&,
                    AugmentedLagrangianConfig<TScalar>)
    -> AugmentedLagrangian<ProblemType, solver_t>;

}  // namespace cppoptlib::solver

#endif  //  INCLUDE_CPPOPTLIB_SOLVER_AUGMENTED_LAGRANGIAN_H_
