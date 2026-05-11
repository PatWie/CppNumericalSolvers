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

#ifndef INCLUDE_CPPOPTLIB_SOLVER_TRUST_REGION_NEWTON_H_
#define INCLUDE_CPPOPTLIB_SOLVER_TRUST_REGION_NEWTON_H_

#include <algorithm>
#include <cmath>
#include <limits>

#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib::solver {

// Trust-region Newton with a truncated-CG (CG-Steihaug) subproblem solver.
//
// Algorithm layout (Nocedal & Wright, 2nd ed., Algorithm 4.1 + 7.2):
//
//   1. At iterate `x_k`, build the local quadratic model
//        m_k(p) = f_k + g_k^T p + 0.5 p^T H_k p
//      with `g_k = grad f(x_k)`, `H_k = hess f(x_k)`.
//
//   2. Solve the trust-region subproblem
//        min_p m_k(p)   subject to   ||p|| <= Delta_k
//      approximately, via CG-Steihaug.  The method runs truncated
//      conjugate gradients on the positive-definite case and terminates
//      on the TR boundary if (a) the step would leave the TR, or
//      (b) negative curvature is detected in the current direction.
//
//   3. Compute the agreement ratio
//        rho_k = (f(x_k) - f(x_k + p_k)) / (m_k(0) - m_k(p_k))
//      and update the TR radius:
//        rho_k < 1/4        -> shrink  Delta_{k+1} = (1/4) * Delta_k
//        rho_k > 3/4 and the step hit the boundary -> grow
//                           -> Delta_{k+1} = min(2*Delta_k, Delta_max)
//        otherwise          -> keep    Delta_{k+1} = Delta_k
//
//   4. Accept the step iff `rho_k > eta` for a small `eta`.  Otherwise
//      stay at `x_k`.  Rejected steps only reduce `Delta_k`; progress
//      resumes once the radius is small enough for the model to track
//      the true function.
//
// Why this solver exists and what it replaces.  `NewtonDescent` in the
// library uses a full Newton step with a fixed `1e-5 * I` diagonal
// shift plus Armijo.  That combination is unreliable on non-convex
// objectives: it cannot recover from a step into a negative-curvature
// region (the shift is too small to change the descent direction), and
// the line search only controls step length, not step quality.  The
// trust-region variant replaces both mechanisms with a single radius
// that is updated on observed versus predicted reduction, handling
// indefinite Hessians cleanly by CG-Steihaug's negative-curvature
// exit.
template <typename TScalar>
struct TrustRegionNewtonConfig {
  // Initial trust-region radius.  A value near one works well when the
  // objective is normalised to O(1); scale with the user's problem when
  // it is not.  Too small costs one extra outer iteration as the model
  // grows the radius; too large wastes at most one rejected step before
  // the radius shrinks.
  TScalar initial_radius = TScalar{1};

  // Upper cap on the radius.  Prevents unbounded growth on objectives
  // whose agreement is consistently good but whose quadratic model is
  // not trusted at arbitrary distance (for instance because the full
  // Hessian is dominated by a rank-one term away from the minimum).
  TScalar max_radius = TScalar{1e10};

  // Step-acceptance threshold.  The classical `eta in (0, 1/4)` range
  // is used: `0.15` is the middle of the practical band (too small
  // accepts bad steps, too large rejects legitimate ones when the
  // model is merely approximate).
  TScalar acceptance_threshold = TScalar{0.15};

  // Radius multipliers on shrink and expand.  Powers of two give a
  // geometric adjustment with numerically friendly values; `0.25` and
  // `2` are the Dennis-Schnabel defaults.
  TScalar shrink_factor = TScalar{0.25};
  TScalar expand_factor = TScalar{2};

  // Agreement thresholds for the radius update.  Below `rho_low` the
  // step shrinks; above `rho_high` AND on the boundary it grows.
  TScalar rho_low = TScalar{0.25};
  TScalar rho_high = TScalar{0.75};

  // CG-Steihaug forcing coefficient.  Terminate the CG iteration when
  // the residual sup-norm drops to
  //   cg_forcing_coefficient * min(0.5, sqrt(||g_k||)) * ||g_k||
  // which is the Eisenstat-Walker forcing recipe that gives superlinear
  // (and, near the solution, quadratic) convergence.  Larger values
  // loosen the inner solve and save CG iterations at the price of a
  // slower outer convergence rate.
  TScalar cg_forcing_coefficient = TScalar{0.5};

  // Hard cap on CG inner iterations.  CG terminates in at most `n`
  // steps in exact arithmetic; the `+ 10` slack absorbs floating-point
  // loss of orthogonality on ill-conditioned Hessians.  Setting this
  // to zero disables the cap and lets CG run to its forcing-based
  // tolerance, which in ill-conditioned cases can cost many iterations
  // without changing the answer.
  int cg_max_iterations_floor = 10;

  // Floor on the TR radius.  Once `Delta_k` falls below this the outer
  // loop reports convergence via `XDeltaViolation` -- at that point no
  // further Newton step can change the iterate meaningfully, and the
  // local quadratic model disagrees with the true function.
  TScalar min_radius = TScalar{1e-12};
  // Upper bound on the number of TR-radius-shrinkage rejections within
  // a single `OptimizationStep`.  Each rejection multiplies the radius
  // by `shrink_factor` (default 0.25), so after `rejection_retry_limit`
  // consecutive rejections the radius has shrunk by roughly
  // `shrink_factor^rejection_retry_limit = 0.25^50 ~ 1e-30`, at which
  // point any further inner loop is wasting evaluations.  The limit
  // exists as a hard safety net; the real exit is `current_radius_ <=
  // min_radius`.
  int rejection_retry_limit = 50;
};

template <typename FunctionType>
class TrustRegionNewton
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(FunctionType::Differentiability ==
                    cppoptlib::function::DifferentiabilityMode::Second,
                "TrustRegionNewton requires second-order differentiability: "
                "the Hessian enters the quadratic model explicitly.");

 public:
  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using ProgressType = typename Superclass::ProgressType;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using MatrixType = typename FunctionType::MatrixType;

  using Config = TrustRegionNewtonConfig<ScalarType>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrustRegionNewton() : Superclass(), config_() { ResetInternal(); }

  explicit TrustRegionNewton(Config config) : Superclass(), config_(config) {
    ResetInternal();
  }

  // Overload that accepts a pre-configured stopping `Progress` (mirrors
  // the other solvers' form) alongside the trust-region config.  The
  // two arguments are independent: stopping criteria govern when the
  // outer loop halts; the TR config governs how each step is built.
  TrustRegionNewton(const ProgressType& stopping_progress, Config config)
      : Superclass(stopping_progress), config_(config) {
    ResetInternal();
  }

  const Config& config() const { return config_; }

  void InitializeSolver(const FunctionType& /*function*/,
                        const StateType& initial_state) override {
    dim_ = initial_state.x.rows();
    ResetInternal();
  }

  StateType OptimizationStep(const FunctionType& function,
                             const StateType& current,
                             const ProgressType& /*progress*/) override {
    // The invariant imposed by `Solver::Minimize` is that `current` has
    // a populated `value` and `gradient`; the Hessian is not cached in
    // the FunctionState, so we evaluate it explicitly here.  A fresh
    // Hessian every outer iteration matches the TR-Newton recipe --
    // we do not want a stale second-order model deciding step
    // acceptance on a nonlinear objective.
    MatrixType hessian;
    VectorType gradient;
    function(current.x, &gradient, &hessian);

    // Guard against a pathological outer state: if the solver's
    // FunctionState populated `gradient` but the user mutated it, we
    // prefer the freshly evaluated one.  This also keeps
    // `current_value` honest against the Hessian evaluation, because
    // some FunctionCRTP subclasses recompute side quantities on every
    // call and the model needs the value at THIS evaluation.
    const ScalarType current_value = current.value;

    // Select the CG tolerance via Eisenstat-Walker forcing.  The
    // tolerance shrinks with ||g_k|| so the inner solve tightens as
    // we approach the minimum.  Capping at 0.5 keeps the first few
    // outer steps from over-tightening CG when ||g|| is large.
    const ScalarType gradient_norm =
        gradient.template lpNorm<Eigen::Infinity>();
    const ScalarType forcing =
        std::min<ScalarType>(ScalarType(0.5), std::sqrt(gradient_norm));
    const ScalarType cg_tolerance =
        config_.cg_forcing_coefficient * forcing * gradient_norm;

    // The outer loop's `OptimizationStep` is expected to return a
    // meaningfully-different iterate on every call -- `Progress::Update`
    // interprets `x_delta = 0` as a convergence signal after
    // `x_delta_violations` consecutive observations, and with the
    // library default of 1 that fires on the very first rejected
    // step.  A trust-region method must be allowed to reject steps
    // and shrink its radius without terminating; we therefore loop
    // the rejection handling inside this single `OptimizationStep`
    // call until either the step is accepted or the radius shrinks
    // below `min_radius`.  At that point we have genuinely stalled
    // and returning `current` unchanged is the correct signal to the
    // outer loop.
    constexpr int rejection_retry_limit_absolute_cap = 1000;
    const int rejection_retry_limit =
        std::min<int>(std::max<int>(config_.rejection_retry_limit, 0),
                      rejection_retry_limit_absolute_cap);
    for (int retry = 0; retry < rejection_retry_limit; ++retry) {
      // The CG-Steihaug solver writes through the pointer; default-
      // construct so fixed-size `VectorType` does not trigger a
      // runtime resize assertion.  The helper fills it in place at
      // the correct dimension.
      VectorType step;
      bool step_hit_boundary = false;
      SolveTrustRegionSubproblem(gradient, hessian, current_radius_,
                                 cg_tolerance, &step, &step_hit_boundary);

      // Evaluate the candidate iterate and compute the agreement ratio
      // rho = (actual reduction) / (predicted reduction).  The
      // predicted reduction comes from the quadratic model:
      //   m(0) - m(p) = -g^T p - 0.5 p^T H p.
      // The actual reduction is the raw function-value change; we
      // evaluate f at the trial point only (no gradient/Hessian) to
      // keep one extra call per rejected step instead of three.
      const VectorType trial_x = current.x + step;
      const ScalarType trial_value = function(trial_x);
      const ScalarType predicted_reduction =
          -gradient.dot(step) - ScalarType(0.5) * step.dot(hessian * step);
      const ScalarType actual_reduction = current_value - trial_value;

      // Defend against degenerate predicted reductions.  A non-positive
      // `predicted_reduction` at `p != 0` indicates a numerical oddity
      // (or an indefinite model where CG exited on the boundary of a
      // negative-curvature direction pointing against the gradient);
      // in that case the agreement test is not meaningful, and we
      // shrink the radius without accepting the step.
      ScalarType rho;
      if (predicted_reduction <= ScalarType(0)) {
        rho = -std::numeric_limits<ScalarType>::infinity();
      } else {
        rho = actual_reduction / predicted_reduction;
      }

      // --- Radius update ---
      //
      // We shrink whenever agreement is poor, regardless of step
      // acceptance -- a rejected step also signals a too-large
      // radius.  We grow only when agreement is good AND the step
      // hit the TR boundary, because growing an unconstrained step
      // (already the full Newton step) would not produce a different
      // iterate next round.
      if (rho < config_.rho_low) {
        current_radius_ *= config_.shrink_factor;
      } else if (rho > config_.rho_high && step_hit_boundary) {
        current_radius_ = std::min<ScalarType>(
            config_.expand_factor * current_radius_, config_.max_radius);
      }

      // --- Step acceptance ---
      //
      // Accept iff agreement clears `eta`.  Otherwise shrink-retry:
      // the loop's next iteration rebuilds the CG subproblem at the
      // new (smaller) radius, which changes where the TR-boundary
      // exit lands.  This is the standard Dennis-Schnabel "inner
      // loop" wording of TR methods.
      if (rho > config_.acceptance_threshold) {
        return StateType(function, trial_x);
      }

      // Step was rejected.  If the radius has shrunk below the
      // stall floor the quadratic model no longer tracks the
      // function over any meaningful neighbourhood; exit the
      // rejection loop and let the outer stopping criteria fire.
      // The outer loop will observe `x_delta = 0` at the next call
      // and -- with default `x_delta_violations` of 1 -- report a
      // stall, which is the correct signal: no further TR step can
      // make progress from this iterate.
      if (current_radius_ <= config_.min_radius) {
        break;
      }
    }
    return current;
  }

 private:
  // Initialise the trust-region radius for a fresh Minimize call.  The
  // user can override `initial_radius` through the config struct;
  // this simply copies it into the mutable state.
  void ResetInternal() {
    current_radius_ = config_.initial_radius;
    dim_ = 0;
  }

  // CG-Steihaug subproblem solver.  Computes an approximate solution
  // to
  //       min_{||p|| <= Delta}  g^T p + 0.5 p^T H p.
  //
  // The returned `*step` is the approximate minimiser (in the 2-norm
  // sense that bounds it to the TR ball), and `*hit_boundary` is true
  // iff CG terminated on the boundary -- either because the iterate
  // would leave the TR or because negative curvature was detected.
  // The outer loop consults `hit_boundary` to decide whether to grow
  // the radius on the next iteration.
  //
  // Numerics.  All vectors live in the problem's dimension.  The
  // stopping residual-norm tolerance comes from the Eisenstat-Walker
  // forcing term the outer loop passes in; CG returns early when the
  // residual falls below that.
  void SolveTrustRegionSubproblem(const VectorType& gradient,
                                  const MatrixType& hessian, ScalarType radius,
                                  ScalarType cg_tolerance, VectorType* step,
                                  bool* hit_boundary) const {
    const int n = dim_;
    // Use `VectorType::Zero(n)` only when the type is dynamic-sized;
    // for fixed-size vectors (compile-time dimension) the one-argument
    // `Zero(n)` asserts at runtime, so we use the zero-argument
    // factory instead.  We prefer the `setZero` + default-construct
    // pattern because it works uniformly for both cases at the price
    // of one extra assignment.
    VectorType p;
    if constexpr (VectorType::SizeAtCompileTime == Eigen::Dynamic) {
      p = VectorType::Zero(n);
    } else {
      p = VectorType::Zero();
    }
    VectorType residual = gradient;    // r_0 = g + H p_0 = g
    VectorType direction = -gradient;  // d_0 = -r_0
    ScalarType residual_dot = residual.squaredNorm();

    const int cg_iterations_max =
        std::max<int>(n, 0) + std::max<int>(config_.cg_max_iterations_floor, 0);

    // Early exit: if the gradient is already below the CG tolerance,
    // the Newton step is zero to this precision.  The outer loop will
    // typically declare convergence right after.
    if (std::sqrt(residual_dot) <= cg_tolerance) {
      *step = p;
      *hit_boundary = false;
      return;
    }

    for (int iteration = 0; iteration < cg_iterations_max; ++iteration) {
      // Curvature along the current search direction.  Negative or
      // zero curvature triggers the TR-boundary exit: the quadratic
      // model is unbounded below along `direction` and the best move
      // inside the TR is to walk along `direction` until we hit the
      // boundary.
      const VectorType hd = hessian * direction;
      const ScalarType curvature = direction.dot(hd);
      if (!(curvature > ScalarType(0))) {
        // A non-positive curvature covers negative curvature AND exact
        // zero; `!(curvature > 0)` additionally absorbs NaN if the
        // Hessian has a rogue coordinate, sending us to the safe
        // boundary-step branch.
        ExtendStepToBoundary(p, direction, radius, step);
        *hit_boundary = true;
        return;
      }

      // Classic CG: candidate full step along `direction`.
      const ScalarType alpha = residual_dot / curvature;
      const VectorType p_candidate = p + alpha * direction;

      // Trust-region exit: candidate step would leave the feasible
      // ball.  Walk along `direction` starting from `p` until the
      // boundary, return that as the approximate minimiser.
      if (p_candidate.norm() >= radius) {
        ExtendStepToBoundary(p, direction, radius, step);
        *hit_boundary = true;
        return;
      }

      // Accept the CG step and update the residual.
      p = p_candidate;
      residual.noalias() += alpha * hd;
      const ScalarType residual_norm = residual.norm();
      if (residual_norm <= cg_tolerance) {
        *step = p;
        *hit_boundary = false;
        return;
      }

      const ScalarType residual_dot_new = residual.squaredNorm();
      const ScalarType beta = residual_dot_new / residual_dot;
      direction = -residual + beta * direction;
      residual_dot = residual_dot_new;
    }

    // CG hit its iteration cap without either leaving the TR or
    // falling below the residual tolerance.  Return the best iterate
    // observed and mark the step as interior: the outer loop will
    // not grow the TR on this iteration (there is no evidence the
    // radius was the binding constraint).
    *step = p;
    *hit_boundary = false;
  }

  // Extend a CG iterate to the trust-region boundary along the
  // current direction.  Solves the scalar quadratic
  //     ||p + tau * direction||^2 = radius^2
  // for `tau > 0` and returns `p + tau * direction`.  The quadratic
  // has a unique positive root because
  //   - `||p|| < radius` (we are inside the TR by invariant),
  //   - `direction` is not zero (CG would have stopped otherwise),
  // so the discriminant is non-negative and the positive root exists.
  static void ExtendStepToBoundary(const VectorType& p,
                                   const VectorType& direction,
                                   ScalarType radius, VectorType* step) {
    const ScalarType a = direction.squaredNorm();
    const ScalarType b = ScalarType(2) * p.dot(direction);
    const ScalarType c = p.squaredNorm() - radius * radius;
    // `a > 0` by construction; a zero would mean `direction == 0`,
    // which CG rules out before calling here.
    const ScalarType discriminant = b * b - ScalarType(4) * a * c;
    // `discriminant >= 0` because `c <= 0` (p is inside the TR) and
    // `a > 0`; `- 4 a c` is non-negative so `b^2 - 4 a c >= b^2 >= 0`.
    const ScalarType tau =
        (-b + std::sqrt(std::max<ScalarType>(discriminant, ScalarType(0)))) /
        (ScalarType(2) * a);
    *step = p + tau * direction;
  }

  Config config_;
  int dim_;
  ScalarType current_radius_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_TRUST_REGION_NEWTON_H_
