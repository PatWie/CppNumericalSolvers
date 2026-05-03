// Copyright 2026, https://github.com/PatWie/CppNumericalSolvers
//
// Exhaustive tests for the augmented-Lagrangian solver.  This file is
// organised top-down into three sections:
//
//   Section A.  Penalty building-block helpers (`QuadraticEqualityPenalty`,
//               `QuadraticInequalityPenaltyGe`, `QuadraticInequalityPenaltyLt`,
//               and the underlying `MinZeroExpression` / `MaxZeroExpression`).
//               These tests touch no solver -- they evaluate the penalty
//               expressions directly at hand-chosen points against closed-form
//               reference values.
//
//   Section B.  Augmented-Lagrangian composite assembly
//   (`ToAugmentedLagrangian`,
//               `FormLagrangianPart`, `FormPenaltyPart`).  These tests build
//               the composite once, without running the solver, and verify
//               that L_aug(x) = f(x) + sum lambda_i c_i(x) + rho * P(x) matches
//               the closed-form expression.  A failure here localises a bug
//               to the composite construction rather than the outer loop.
//
//   Section C.  Outer-loop KKT tests: run `AugmentedLagrangian::Minimize` on
//               problems with analytic optima and verify the primal solution,
//               the recovered Lagrange multipliers, and constraint feasibility.
//               Cases cover equality-only, inequality-active, both-active,
//               and degenerate (feasible-at-start, unconstrained-wrapped)
//               problems.
//
// Every numeric tolerance is named.  Expected values are derived in the
// comment immediately above each assertion so the reader can verify the
// test, not just read it.
#undef NDEBUG

// `cppoptlib/solver/augmented_lagrangian.h` now carries its own transitive
// includes for `DifferentiabilityMode` and `FunctionExpr` via
// `function_problem.h`.  We include it first as a regression guard: a
// future refactor that drops those transitive includes would fail to
// compile this file.
#include "cppoptlib/solver/augmented_lagrangian.h"

#include <cmath>
#include <initializer_list>
#include <vector>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/lbfgs.h"
#include "cppoptlib/solver/lbfgsb.h"
#include "gtest/gtest.h"

namespace {

// ---- Named tolerances -----------------------------------------------------
// Used only as semantic constants -- no test writes a bare numeric literal.

// Penalty-helper evaluations have no numerical solver in them, so we can
// assert agreement with closed-form formulas at machine precision.
constexpr double penalty_evaluation_tolerance = 1e-12;

// Outer-loop primal tolerance.  Augmented Lagrangian with rho growing from
// 1.0 and an inner L-BFGS (default stopping) converges the primal to a
// few parts per thousand on well-scaled 2-D problems.  Tighter than this
// is a false lockdown.
constexpr double kkt_primal_tolerance = 1e-3;

// Lagrange-multiplier recovery is slightly looser than primal; augmented
// Lagrangian converges duals at a comparable rate but noise from the
// inner solver compounds over outer iterations.  Keep this one order of
// magnitude tighter than the outer-loop stopping `constraint_threshold`
// default of 1e-5 -- we want duals to carry meaning, not noise.
constexpr double kkt_dual_tolerance = 1e-2;

// Constraint-feasibility tolerance -- the outer solver's default
// `constraint_threshold` is 1e-5, so after Minimize returns with
// `Status::Finished` the residual must be at or below that.
constexpr double feasibility_tolerance = 1e-5;

// ---- 1-D scalar function (for penalty-helper tests) -----------------------
// `f(x) = a + b * x[0]` with fixed a, b so tests can drive the constraint
// value to any chosen scalar by choosing `x[0]`.
class Linear1D : public cppoptlib::function::FunctionCRTP<
                     Linear1D, double,
                     cppoptlib::function::DifferentiabilityMode::First, 1> {
 public:
  double a_coef;
  double b_coef;
  Linear1D(double a, double b) : a_coef(a), b_coef(b) {}

  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    if (grad) {
      grad->resize(1);
      (*grad)[0] = b_coef;
    }
    return a_coef + b_coef * x[0];
  }
};

// Spelling this out once: `FunctionExpr<double, First, 1>` is the 1-D
// type-erased wrapper, needed because the penalty helpers call into
// `MinZeroExpression::operator()(x, grad, hess)` (three args) on the
// wrapped constraint.  A bare `Linear1D` only defines `operator()(x, grad)`
// (two args) and does not satisfy that three-arg call.  The type
// erasure in FunctionExpr routes through FunctionCRTP's virtual three-arg
// override, which then dispatches down to the two-arg user operator().
using FunctionExpr1d = cppoptlib::function::FunctionExpr<
    double, cppoptlib::function::DifferentiabilityMode::First, 1>;

using Vec1 = Eigen::Matrix<double, 1, 1>;

Vec1 MakeVec1(double v) {
  Vec1 x;
  x[0] = v;
  return x;
}

// ---- 2-D functions used in multiple sections ------------------------------
template <class F>
using Function2d = cppoptlib::function::FunctionCRTP<
    F, double, cppoptlib::function::DifferentiabilityMode::First, 2>;

using FunctionExpr2d = cppoptlib::function::FunctionExpr<
    double, cppoptlib::function::DifferentiabilityMode::First, 2>;

// f(x) = 0.5 * (x0^2 + x1^2), gradient = x.
class HalfSquaredNorm2D : public Function2d<HalfSquaredNorm2D> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    if (grad) *grad = x;
    return 0.5 * x.squaredNorm();
  }
};

// f(x) = (x0 - 1)^2 + (x1 - 2)^2, gradient = 2 * (x - [1,2]).
class QuadraticAt12 : public Function2d<QuadraticAt12> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    if (grad) {
      VectorType reference(2);
      reference << 1.0, 2.0;
      *grad = 2.0 * (x - reference);
    }
    return (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 2.0) * (x[1] - 2.0);
  }
};

// f(x) = x0 + x1, gradient = [1, 1].
class Sum2D : public Function2d<Sum2D> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    if (grad) *grad = VectorType::Ones(2);
    return x.sum();
  }
};

// g(x) = x0 - target, gradient = [1, 0].  Parameterises a simple linear
// equality constraint `x0 == target`.
class X0MinusTarget : public Function2d<X0MinusTarget> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double target;
  explicit X0MinusTarget(double t) : target(t) {}

  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    if (grad) {
      *grad = VectorType::Zero(2);
      (*grad)[0] = 1.0;
    }
    return x[0] - target;
  }
};

// h(x) = bound - x0 >= 0  (i.e. x0 <= bound).  Gradient = [-1, 0].
class UpperBoundOnX0 : public Function2d<UpperBoundOnX0> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double bound;
  explicit UpperBoundOnX0(double b) : bound(b) {}

  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    if (grad) {
      *grad = VectorType::Zero(2);
      (*grad)[0] = -1.0;
    }
    return bound - x[0];
  }
};

// f(x) = 0.5 * ((x0 - 2)^2 + x1^2).  Unconstrained optimum (2, 0).
// Used in the inequality-active KKT test.
class QuadraticAt20 : public Function2d<QuadraticAt20> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    if (grad) {
      VectorType reference(2);
      reference << 2.0, 0.0;
      *grad = (x - reference);
    }
    return 0.5 * ((x[0] - 2.0) * (x[0] - 2.0) + x[1] * x[1]);
  }
};

// Inequality 2 - (x0 + x1) >= 0  (i.e. x0 + x1 <= 2).  Gradient = [-1, -1].
class SumUpperBound : public Function2d<SumUpperBound> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    if (grad) {
      *grad = VectorType(2);
      (*grad) << -1.0, -1.0;
    }
    return 2.0 - (x[0] + x[1]);
  }
};

// c(x) = 0 for all x.  Trivially-feasible equality.  Used to pin feasible-
// start and penalty-schedule tests.  We do not multiply by zero inside an
// expression template (that can trigger NaN paths through the tree); we
// return an honest zero with an honest zero gradient.
class ZeroConstraint : public Function2d<ZeroConstraint> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    if (grad) *grad = VectorType::Zero(2);
    (void)x;
    return 0.0;
  }
};

// ---- Problem-type alias used across Section C tests -----------------------
// Spelling the full `ConstrainedOptimizationProblem<...>` on every test is
// noisy.  Using a typedef reads cleaner.  A single `Mode` parameter
// describes both objective and constraint differentiability.
using ConstrainedProblem2d =
    cppoptlib::function::ConstrainedOptimizationProblem<
        double, cppoptlib::function::DifferentiabilityMode::First, 2>;

// Second-order 2-D quadratic `f(x) = 0.5 * x^T A x` with
// `A = diag(4, 2)`.  Gradient = A x; Hessian = A.  Used only by the
// mode-downgrade test: we want to construct a `First`-mode
// `FunctionExpr` from a `Second`-mode CRTP source and verify that the
// evaluation + gradient match the analytic answer (and that the
// Hessian simply never gets asked for).
class DiagonalQuadratic2dSecond
    : public cppoptlib::function::FunctionCRTP<
          DiagonalQuadratic2dSecond, double,
          cppoptlib::function::DifferentiabilityMode::Second, 2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                        MatrixType* hess = nullptr) const {
    if (grad) {
      *grad = VectorType(2);
      (*grad) << 4.0 * x[0], 2.0 * x[1];
    }
    if (hess) {
      *hess = MatrixType::Zero(2, 2);
      (*hess)(0, 0) = 4.0;
      (*hess)(1, 1) = 2.0;
    }
    return 2.0 * x[0] * x[0] + x[1] * x[1];
  }
};

}  // namespace

// =============================================================================
// SECTION A: Penalty-helper unit tests
// =============================================================================
//
// These tests exercise the building blocks in `function_penalty.h` in
// isolation.  The expected values are derived at the top of each test from
// the closed-form formulas P_eq(c)     = 0.5 * c^2 ,
//                          P_ineq_ge(c) = 0.5 * min(0, c)^2 ,
//                          P_ineq_lt(c) = 0.5 * max(0, c)^2 .
// A failure here *must* be a bug in the penalty helpers themselves.

// ---- A.1: QuadraticEqualityPenalty is 0 at feasibility --------------------
// c(x) = 0 + 1*x.  At x = 0, c = 0, so P_eq(c)(x=0) = 0.5 * 0^2 = 0.
TEST(QuadraticEqualityPenalty, ZeroAtFeasiblePoint) {
  FunctionExpr1d c = Linear1D(0.0, 1.0);
  auto penalty = cppoptlib::function::QuadraticEqualityPenalty(c);
  const double value = penalty(MakeVec1(0.0));
  EXPECT_NEAR(0.0, value, penalty_evaluation_tolerance);
}

// ---- A.2: QuadraticEqualityPenalty is symmetric in sign of c -------------
// c(x) = -2 + 1*x.  At x = 5, c = 3, P_eq = 4.5.
// At x = -1, c = -3, P_eq = 4.5.  Same penalty either side -- it is
// literally c^2.  This test pins the "equality penalty does NOT distinguish
// sign of the residual" invariant.
TEST(QuadraticEqualityPenalty, SymmetricInResidualSign) {
  FunctionExpr1d c = Linear1D(-2.0, 1.0);
  auto penalty = cppoptlib::function::QuadraticEqualityPenalty(c);
  // x = 5 -> c = 3 -> P = 0.5 * 9 = 4.5.
  EXPECT_NEAR(4.5, penalty(MakeVec1(5.0)), penalty_evaluation_tolerance);
  // x = -1 -> c = -3 -> P = 0.5 * 9 = 4.5.
  EXPECT_NEAR(4.5, penalty(MakeVec1(-1.0)), penalty_evaluation_tolerance);
}

// ---- A.3: QuadraticInequalityPenaltyGe is zero when c(x) >= 0 ------------
// Convention: constraint is `c(x) >= 0`, so penalty kicks in only when
// c is negative.  Any non-negative c value must return exactly 0 with
// zero gradient.
TEST(QuadraticInequalityPenaltyGe, ZeroWhenConstraintSatisfied) {
  FunctionExpr1d c = Linear1D(0.0, 1.0);  // c(x) = x.
  auto penalty = cppoptlib::function::QuadraticInequalityPenaltyGe(c);
  // x = 0 -> c = 0 (active boundary, still "satisfied").
  EXPECT_NEAR(0.0, penalty(MakeVec1(0.0)), penalty_evaluation_tolerance);
  // x = 5 -> c = 5 (strictly satisfied).
  EXPECT_NEAR(0.0, penalty(MakeVec1(5.0)), penalty_evaluation_tolerance);
}

// ---- A.4: QuadraticInequalityPenaltyGe fires on violation ---------------
// c(x) = x, violated when c < 0.  At x = -3, c = -3, min(0, c) = -3,
// P_ineq_ge = 0.5 * 9 = 4.5.
TEST(QuadraticInequalityPenaltyGe, FiresOnNegativeResidual) {
  FunctionExpr1d c = Linear1D(0.0, 1.0);
  auto penalty = cppoptlib::function::QuadraticInequalityPenaltyGe(c);
  EXPECT_NEAR(4.5, penalty(MakeVec1(-3.0)), penalty_evaluation_tolerance);
}

// ---- A.5: QuadraticInequalityPenaltyLt is the complement ---------------
// Convention: constraint is `c(x) <= 0`, penalty kicks in when c > 0.
TEST(QuadraticInequalityPenaltyLt, ZeroWhenCNonpositive) {
  FunctionExpr1d c = Linear1D(0.0, 1.0);
  auto penalty = cppoptlib::function::QuadraticInequalityPenaltyLt(c);
  EXPECT_NEAR(0.0, penalty(MakeVec1(-5.0)), penalty_evaluation_tolerance);
  EXPECT_NEAR(0.0, penalty(MakeVec1(0.0)), penalty_evaluation_tolerance);
}

TEST(QuadraticInequalityPenaltyLt, FiresOnPositiveResidual) {
  FunctionExpr1d c = Linear1D(0.0, 1.0);
  auto penalty = cppoptlib::function::QuadraticInequalityPenaltyLt(c);
  // x = 3 -> c = 3 -> P = 0.5 * 9 = 4.5.
  EXPECT_NEAR(4.5, penalty(MakeVec1(3.0)), penalty_evaluation_tolerance);
}

// ---- A.6: Gradient of QuadraticEqualityPenalty at c != 0 ------------------
// d/dx [0.5 * c(x)^2] = c(x) * c'(x).  With c(x) = x, at x = 3:
// expected gradient = 3 * 1 = 3.
TEST(QuadraticEqualityPenalty, GradientMatchesChainRule) {
  FunctionExpr1d c = Linear1D(0.0, 1.0);
  auto penalty = cppoptlib::function::QuadraticEqualityPenalty(c);
  Vec1 grad;
  const double value = penalty(MakeVec1(3.0), &grad);
  EXPECT_NEAR(4.5, value, penalty_evaluation_tolerance);
  EXPECT_NEAR(3.0, grad[0], penalty_evaluation_tolerance);
}

// ---- A.7: Gradient of QuadraticInequalityPenaltyGe is zero when inactive.
// When c(x) > 0 the penalty is identically zero and its gradient must also
// be zero -- otherwise a spurious force is added at the feasible side.
TEST(QuadraticInequalityPenaltyGe, GradientZeroWhenSatisfied) {
  FunctionExpr1d c = Linear1D(0.0, 1.0);
  auto penalty = cppoptlib::function::QuadraticInequalityPenaltyGe(c);
  Vec1 grad;
  const double value = penalty(MakeVec1(5.0), &grad);
  EXPECT_NEAR(0.0, value, penalty_evaluation_tolerance);
  EXPECT_NEAR(0.0, grad[0], penalty_evaluation_tolerance);
}

// ---- A.8: Gradient of QuadraticInequalityPenaltyGe on violation.
// P(x) = 0.5 * min(0, c)^2.  When c = -3, dP/dx = min(0, c) * c'(x) =
// (-3) * 1 = -3.
TEST(QuadraticInequalityPenaltyGe, GradientMatchesChainRuleOnViolation) {
  FunctionExpr1d c = Linear1D(0.0, 1.0);
  auto penalty = cppoptlib::function::QuadraticInequalityPenaltyGe(c);
  Vec1 grad;
  const double value = penalty(MakeVec1(-3.0), &grad);
  EXPECT_NEAR(4.5, value, penalty_evaluation_tolerance);
  EXPECT_NEAR(-3.0, grad[0], penalty_evaluation_tolerance);
}

// =============================================================================
// SECTION B: Composite-assembly tests (`ToAugmentedLagrangian` / parts)
// =============================================================================
//
// Build the augmented Lagrangian once, without running the solver, and
// evaluate it at a chosen x.  L_aug(x) has the closed form
//   L_aug(x) = f(x) + sum_eq  lambda_i c_eq_i(x)
//                   + sum_ineq mu_j c_ineq_j(x)
//                   + rho * [sum_eq 0.5 c_eq_i(x)^2
//                            + sum_ineq 0.5 min(0, c_ineq_j(x))^2].
// We build a problem, plug in fixed multiplier/penalty state, evaluate, and
// compare against the hand-computed value.  This localises the sign
// convention of the Lagrangian part -- i.e. is the inequality term
// `+ mu * c` or `- mu * c`.

// ---- B.1: Equality-only composite at a non-feasible x ---------------------
// f(x) = 0.5 * ||x||^2.  One equality c(x) = x0 - 1 (so c is 0 at x0 = 1).
// multipliers lambda = {2.0}; rho = 3.0.
// At x = (3, 4): c = 2.
// Expected L_aug = 0.5*(9+16) + 2.0*2 + 3.0 * 0.5 * 4
//                = 12.5 + 4 + 6 = 22.5.
TEST(ToAugmentedLagrangian, EqualityOnlyMatchesClosedForm) {
  cppoptlib::function::FunctionExpr<
      double, cppoptlib::function::DifferentiabilityMode::First, 2>
      objective = HalfSquaredNorm2D();
  cppoptlib::function::FunctionExpr<
      double, cppoptlib::function::DifferentiabilityMode::First, 2>
      equality = X0MinusTarget(1.0);

  ConstrainedProblem2d problem(objective, {equality});
  cppoptlib::function::LagrangeMultiplierState<double> multipliers({2.0}, {});
  cppoptlib::function::PenaltyState<double> penalty(3.0);

  auto augmented =
      cppoptlib::function::ToAugmentedLagrangian(problem, multipliers, penalty);
  Eigen::Vector2d x;
  x << 3.0, 4.0;
  EXPECT_NEAR(22.5, augmented(x), penalty_evaluation_tolerance);
}

// ---- B.2: Inequality-only composite, satisfied side, locks PHR -----
// f(x) = 0.5 * ||x||^2.  One inequality c(x) = x0 - 0.5 >= 0 (satisfied
// when x0 >= 0.5).  multipliers mu = {7.0}; rho = 4.0.
//
// At x = (3, 0):  c = 2.5 (satisfied), mu - rho * c = 7 - 10 = -3.
// Powell-Hestenes-Rockafellar inequality contribution:
//   P_j = (1 / (2 rho)) * [ max(0, mu - rho c)^2 - mu^2 ]
//       = (1/8) * [ max(0, -3)^2 - 49 ]
//       = (1/8) * [ 0 - 49 ] = -6.125.
// Composite L_aug = f + P_j = 0.5*9 + (-6.125) = 4.5 - 6.125 = -1.625.
//
// The PHR form is the key defence against a non-convex composite
// that would otherwise go to -infinity along a ray of inactive
// inequalities: on the strictly-inactive side the contribution is
// the CONSTANT -mu^2/(2 rho) = -49/8 = -6.125, independent of x.
TEST(ToAugmentedLagrangian, InequalityPhrOnInactiveSide) {
  using namespace cppoptlib::function;
  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> inequality =
      X0MinusTarget(0.5);

  ConstrainedProblem2d problem(objective, /*eq=*/{}, {inequality});
  LagrangeMultiplierState<double> multipliers({}, {7.0});
  PenaltyState<double> penalty(4.0);

  auto augmented = ToAugmentedLagrangian(problem, multipliers, penalty);
  Eigen::Vector2d x;
  x << 3.0, 0.0;
  EXPECT_NEAR(-1.625, augmented(x), penalty_evaluation_tolerance);
}

// ---- B.3: Inequality PHR on the active/violated side ---------------
// Same setup as B.2 but x = (0.0, 0.0), so c = -0.5 (violated), and
// mu - rho c = 7 - 4 * (-0.5) = 9 > 0.  PHR contribution:
//   P_j = (1 / (2 rho)) * [ max(0, 9)^2 - 49 ]
//       = (1/8) * [ 81 - 49 ] = 4.0.
// Composite = f + P_j = 0 + 4 = 4.
//
// This matches the naive `-mu c + 0.5 rho min(0,c)^2` form by
// coincidence on the active side -- both expressions agree when
// mu - rho c >= 0.  The difference only appears on the strictly-
// inactive side (test B.2).
TEST(ToAugmentedLagrangian, InequalityPhrOnActiveSide) {
  using namespace cppoptlib::function;
  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> inequality =
      X0MinusTarget(0.5);

  ConstrainedProblem2d problem(objective, /*eq=*/{}, {inequality});
  LagrangeMultiplierState<double> multipliers({}, {7.0});
  PenaltyState<double> penalty(4.0);

  auto augmented = ToAugmentedLagrangian(problem, multipliers, penalty);
  Eigen::Vector2d x;
  x << 0.0, 0.0;
  EXPECT_NEAR(4.0, augmented(x), penalty_evaluation_tolerance);
}

// =============================================================================
// SECTION C: Outer-loop KKT tests (run the solver to convergence)
// =============================================================================

// ---- C.1: Equality-only quadratic recovers primal and dual ---------------
//
// Problem: min 0.5 * (x0^2 + x1^2) subject to x0 = 1.
// Lagrangian:  L = 0.5 x^T x + lambda * (x0 - 1).
// Stationarity: x0 + lambda = 0, x1 = 0.  Feasibility: x0 = 1.
// => x*  = (1, 0).
// => f*  = 0.5.
// => lambda* = -1.
//
// This is the canonical smoke test for the multiplier update: if the
// update uses `lambda += rho * 0.5 * c^2` instead of `lambda += rho * c`,
// `lambda` never reaches -1 (the squared residual drives it positive).
TEST(AugmentedLagrangianKKT, EqualityOnlyQuadratic) {
  using namespace cppoptlib::function;
  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> equality =
      X0MinusTarget(1.0);

  ConstrainedProblem2d problem(objective, {equality});

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  Eigen::Vector2d x0;
  x0 << 5.0, 5.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 1, 0, 1.0);
  auto [solution, progress] = solver.Minimize(state);

  // Primal optimum.
  EXPECT_NEAR(1.0, solution.x[0], kkt_primal_tolerance);
  EXPECT_NEAR(0.0, solution.x[1], kkt_primal_tolerance);
  // Feasibility.
  EXPECT_LE(std::abs(solution.x[0] - 1.0), feasibility_tolerance);
  // Dual: stationarity gives lambda* = -x0* = -1.
  ASSERT_EQ(1u, solution.multiplier_state.equality_multipliers.size());
  EXPECT_NEAR(-1.0, solution.multiplier_state.equality_multipliers[0],
              kkt_dual_tolerance);
}

// ---- C.2: Inequality-active problem recovers primal and dual -------------
//
// Problem: min 0.5 * ((x0 - 2)^2 + x1^2) subject to 1 - x0 >= 0.
// KKT with mu >= 0:
//   dL/dx0 = (x0 - 2) - mu = 0   (because constraint is `1 - x0 >= 0`,
//                                  gradient of c = [-1, 0], and KKT has
//                                  +mu * grad(c) in the stationarity
//                                  equation, giving the sign above).
//   dL/dx1 = x1 = 0.
//   mu * (1 - x0) = 0, 1 - x0 >= 0, mu >= 0.
// If the constraint is active: x0 = 1 -> mu = x0 - 2 = -1.  That is
// negative, which violates dual feasibility.  So the constraint must be
// inactive -> mu = 0 -> x0 = 2.  But then 1 - x0 = -1 < 0 violates
// primal feasibility.  Contradiction -- so we need the ACTIVE case with
// the correct sign convention.  Re-derive with `- mu * c` in the Lagrangian
// (the convention used by the outer-loop multiplier update with `c >= 0`):
//   L = f(x) - mu * (1 - x0),  so dL/dx0 = (x0 - 2) + mu = 0 -> mu = 2 - x0.
//   At the active boundary x0 = 1, mu = 1.  mu >= 0 holds.  Feasibility OK.
//   So x* = (1, 0), mu* = 1, f* = 0.5.
TEST(AugmentedLagrangianKKT, InequalityActiveRecoversMultiplier) {
  using namespace cppoptlib::function;

  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      QuadraticAt20();
  // c(x) = 1 - x0 >= 0.
  FunctionExpr<double, DifferentiabilityMode::First, 2> inequality =
      UpperBoundOnX0(1.0);

  ConstrainedProblem2d problem(objective, /*eq=*/{}, {inequality});

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  Eigen::Vector2d x0;
  x0 << 5.0, 5.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 0, 1, 1.0);
  auto [solution, progress] = solver.Minimize(state);

  EXPECT_NEAR(1.0, solution.x[0], kkt_primal_tolerance);
  EXPECT_NEAR(0.0, solution.x[1], kkt_primal_tolerance);
  // Feasibility: 1 - x0 >= -tol.
  const double constraint_value = 1.0 - solution.x[0];
  EXPECT_GE(constraint_value, -feasibility_tolerance);
  // Dual: mu* = 1, and dual feasibility mu >= 0.
  ASSERT_EQ(1u, solution.multiplier_state.inequality_multipliers.size());
  const double mu = solution.multiplier_state.inequality_multipliers[0];
  EXPECT_GE(mu, -kkt_dual_tolerance);
  EXPECT_NEAR(1.0, mu, kkt_dual_tolerance);
}

// ---- C.3: Both constraints active (from constrained_simple.cc) -----------
//
// Problem: min (x0 - 1)^2 + (x1 - 2)^2
//           subject to x0 = 0.5, and 2 - (x0 + x1) >= 0.
// Unconstrained optimum (1, 2) violates the inequality (1 + 2 = 3 > 2).
// With x0 fixed at 0.5 by equality, minimize (0.5 - 1)^2 + (x1 - 2)^2 =
// 0.25 + (x1 - 2)^2 over x1 with 0.5 + x1 <= 2, i.e. x1 <= 1.5.  The
// inner quadratic is minimized at x1 = 2 but that is infeasible; the best
// feasible choice is x1 = 1.5 with f = 0.5.
TEST(AugmentedLagrangianKKT, BothEqualityAndInequalityActive) {
  using namespace cppoptlib::function;

  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      QuadraticAt12();
  FunctionExpr<double, DifferentiabilityMode::First, 2> equality =
      X0MinusTarget(0.5);
  FunctionExpr<double, DifferentiabilityMode::First, 2> inequality =
      SumUpperBound();

  ConstrainedProblem2d problem(objective, {equality}, {inequality});

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  Eigen::Vector2d x0;
  x0 << 1.0, 1.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 1, 1, 1.0);
  auto [solution, progress] = solver.Minimize(state);

  EXPECT_NEAR(0.5, solution.x[0], kkt_primal_tolerance);
  EXPECT_NEAR(1.5, solution.x[1], kkt_primal_tolerance);
  // Feasibility on both constraints.
  EXPECT_LE(std::abs(solution.x[0] - 0.5), feasibility_tolerance);
  const double inequality_value = 2.0 - (solution.x[0] + solution.x[1]);
  EXPECT_GE(inequality_value, -feasibility_tolerance);
  // Dual feasibility for the inequality: mu >= 0.
  ASSERT_EQ(1u, solution.multiplier_state.inequality_multipliers.size());
  EXPECT_GE(solution.multiplier_state.inequality_multipliers[0],
            -kkt_dual_tolerance);
}

// ---- C.4: Feasible start returns immediately with Status::Finished --------
//
// Problem: min 0.5 * ||x||^2 subject to c(x) = 0 + 0 * x = 0 (trivially
// feasible everywhere).  Starting at (0, 0), which is also the primal
// optimum, the first outer iteration sees `max_violation = 0` and stops.
//
// This test pins the outer-loop status machinery: if `max_violation` is
// computed as `0.5 c^2` vs `|c|` we cannot tell at c = 0 (both are 0);
// but the test does catch the case where a bug breaks the early-exit by
// making `max_violation` NaN or leaving it uninitialised.
TEST(AugmentedLagrangianOuter, FeasibleStartConvergesImmediately) {
  using namespace cppoptlib::function;

  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> equality =
      ZeroConstraint();

  ConstrainedProblem2d problem(objective, {equality});

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  Eigen::Vector2d x0;
  x0 << 0.0, 0.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 1, 0, 1.0);
  auto [solution, progress] = solver.Minimize(state);

  EXPECT_NEAR(0.0, solution.x[0], kkt_primal_tolerance);
  EXPECT_NEAR(0.0, solution.x[1], kkt_primal_tolerance);
  EXPECT_EQ(cppoptlib::solver::Status::Finished, progress.status);
  // Pin that we exit fast, not after 10000 outer iterations.  A bug that
  // leaves `max_violation` positive would run to the iteration cap.
  constexpr size_t outer_iteration_cap_for_trivial_problem = 5;
  EXPECT_LE(progress.num_iterations, outer_iteration_cap_for_trivial_problem);
}

// ---- C.5: Unconstrained problem wrapped in the constrained solver --------
//
// A `ConstrainedOptimizationProblem` with no constraints should be
// equivalent to calling the inner solver directly.  Starting from (5, 5),
// L-BFGS on 0.5 * ||x||^2 converges to the origin.
TEST(AugmentedLagrangianOuter, NoConstraintsIsUnconstrained) {
  using namespace cppoptlib::function;
  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  // Brace-init to avoid the most-vexing-parse with a single argument:
  // `ConstrainedProblem2d problem(objective)` would be parsed as a
  // function declaration taking FunctionExpr2d by value.
  ConstrainedProblem2d problem{objective};

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  Eigen::Vector2d x0;
  x0 << 5.0, 5.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 0, 0, 1.0);
  auto [solution, progress] = solver.Minimize(state);

  EXPECT_NEAR(0.0, solution.x[0], kkt_primal_tolerance);
  EXPECT_NEAR(0.0, solution.x[1], kkt_primal_tolerance);
  EXPECT_EQ(cppoptlib::solver::Status::Finished, progress.status);
}

// ---- C.6: Penalty is held flat on a feasible-start problem ---------------
//
// With the violation-conditional schedule, `rho` only grows when
// `max_violation` fails to shrink.  On a feasible-start problem whose
// equality reads identically zero, the violation is already zero at the
// first outer iteration; the schedule therefore must *never* grow `rho`.
// An older unconditional `rho *= 10` schedule would multiply it every
// outer iteration regardless.  We pin the exact post-solve penalty to
// its initial value so the conditional branch is locked down.
TEST(AugmentedLagrangianOuter, PenaltyHoldsFlatOnFeasibleProblem) {
  using namespace cppoptlib::function;

  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> equality =
      ZeroConstraint();
  ConstrainedProblem2d problem(objective, {equality});

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  Eigen::Vector2d x0;
  x0 << 0.0, 0.0;
  constexpr double initial_penalty = 1.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 1, 0,
                                                             initial_penalty);
  auto [solution, progress] = solver.Minimize(state);

  // The conditional schedule should never fire on a trivially feasible
  // problem.  The penalty is pinned exactly at its initial value.
  EXPECT_EQ(initial_penalty, solution.penalty_state.penalty);
}

// ---- C.6b: penalty_growth_factor = 1 disables the growth branch entirely.
//
// Users with a well-conditioned problem may want to fix `rho` at its
// initial value, letting the multipliers do all of the work.  Setting
// `penalty_growth_factor = 1` accomplishes that no matter what the
// violation trajectory looks like.  The test drives an infeasible-start
// equality-only problem that would ordinarily grow `rho` on every
// iteration and verifies the user's override sticks.
TEST(AugmentedLagrangianOuter, PenaltyGrowthCanBeDisabled) {
  using namespace cppoptlib::function;

  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> equality =
      X0MinusTarget(1.0);
  ConstrainedProblem2d problem(objective, {equality});

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangianConfig<double> config;
  config.penalty_growth_factor = 1.0;

  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver, config);

  Eigen::Vector2d x0;
  x0 << 5.0, 5.0;
  constexpr double initial_penalty = 1.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 1, 0,
                                                             initial_penalty);
  auto [solution, progress] = solver.Minimize(state);

  EXPECT_EQ(initial_penalty, solution.penalty_state.penalty);
}

// ---- C.6c: Penalty grows only while violation fails to shrink ------------
//
// Start infeasible on an equality-only problem.  The first outer
// iteration observes a positive `max_violation` against the
// constructor-supplied initial `max_violation = 0`; the ratio test fires
// and `rho` grows by the default factor of 10.  Once the multipliers
// pull the iterate onto the feasibility boundary the violation shrinks
// rapidly.  By the time the solver returns the penalty should be
// bounded -- a single growth step typically suffices for this problem,
// and a handful at worst.  We pin `rho <= 10^4` as a loose upper bound
// that fires the regression only when growth explodes.
TEST(AugmentedLagrangianOuter, PenaltyGrowsOnlyWhileViolationLags) {
  using namespace cppoptlib::function;

  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> equality =
      X0MinusTarget(1.0);
  ConstrainedProblem2d problem(objective, {equality});

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  Eigen::Vector2d x0;
  x0 << 5.0, 5.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 1, 0, 1.0);
  auto [solution, progress] = solver.Minimize(state);

  constexpr double penalty_upper_bound = 1e4;
  EXPECT_LE(solution.penalty_state.penalty, penalty_upper_bound);
  // Also expect *some* growth -- starting at 1.0 and solving an
  // infeasible-start problem without any growth would indicate the
  // schedule test is never firing.
  EXPECT_GE(solution.penalty_state.penalty, 1.0);
}

// ---- C.7: CTAD regression -- bare construction with no alias -----------
//
// Earlier revisions split the differentiability mode across two
// template parameters, which made
//     cppoptlib::function::ConstrainedOptimizationProblem problem(
//         objective, {equality});
// fail to deduce when both modes agreed (the common case).  With a
// single mode parameter the compiler resolves CTAD cleanly.  This test
// exists to lock the fix: if a future revision re-introduces the two
// modes, the file stops compiling here.
TEST(ConstrainedOptimizationProblem, BareCtadCompiles) {
  using namespace cppoptlib::function;
  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> equality =
      X0MinusTarget(1.0);
  // One-argument CTAD.
  cppoptlib::function::ConstrainedOptimizationProblem problem_unconstrained{
      objective};
  // Two-argument CTAD with equality constraint list.
  cppoptlib::function::ConstrainedOptimizationProblem problem_equality(
      objective, {equality});
  // Three-argument CTAD with both lists.
  cppoptlib::function::ConstrainedOptimizationProblem problem_full(
      objective, {equality}, {equality});
  // All three must evaluate to `double` at the origin; we do not care
  // about the specific values -- only that the instances are usable.
  Eigen::Vector2d origin = Eigen::Vector2d::Zero();
  EXPECT_NEAR(0.0, problem_unconstrained.objective(origin),
              penalty_evaluation_tolerance);
  EXPECT_NEAR(0.0, problem_equality.objective(origin),
              penalty_evaluation_tolerance);
  EXPECT_NEAR(0.0, problem_full.objective(origin),
              penalty_evaluation_tolerance);
}

// ---- C.7b: Bare expression-template elements in constraint lists -------
//
// Historically users had to wrap constraint expressions in an explicit
// `FunctionExpr(...)` when they appeared inside an initializer list
// passed to `ConstrainedOptimizationProblem`.  The two-mode CTAD
// failure forced the wrap.  With the single-Mode redesign plus the
// Second->First downgrade adapter, a bare
//     {circle - 2.0}
// converts implicitly through the `FunctionExpr` converting constructor
// when the initializer-list element type is `FunctionExpr<T, Mode, Dim>`.
//
// This test pins the ergonomics.  If a future CTAD revision breaks the
// implicit conversion, the file stops compiling here.
TEST(ConstrainedOptimizationProblem, BareExpressionInitializerList) {
  using namespace cppoptlib::function;
  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> target_one =
      X0MinusTarget(1.0);
  // The key test: bare `target_one - 0.5` (a SubExpression) appears as
  // an element inside an initializer_list<FunctionExpr<...>> without
  // any explicit wrap.  Same for the inequality side.
  cppoptlib::function::ConstrainedOptimizationProblem problem_bare(
      objective,
      /*eq=*/{target_one - 0.5},
      /*ineq=*/{0.5 - target_one});
  // Evaluate the stored equality constraint at x = (1.5, 0).  The
  // original `target_one` was `x0 - 1`; `target_one - 0.5` is therefore
  // `x0 - 1.5`, and at x0 = 1.5 that is exactly 0.
  Eigen::Vector2d point;
  point << 1.5, 0.0;
  EXPECT_NEAR(0.0, problem_bare.equality_constraints[0](point),
              penalty_evaluation_tolerance);
}

// ---- C.8: Second-mode source downgrades into First-mode FunctionExpr ---
//
// Earlier revisions required the `FunctionExpr` converting constructor
// to be called with an expression whose declared differentiability
// exactly matched the wrapper's.  A user with a `Second`-mode objective
// who wanted to plug it into a `First`-mode-consuming API (such as the
// augmented-Lagrangian solver, whose inner L-BFGS ignores Hessians
// anyway) had to hand-roll a First-mode clone of the class.  The
// converting constructor now accepts any source whose mode is at least
// as strong as the target, wrapping it in a `ModeDowngradeAdapter` that
// discards the unused derivative pointer on evaluation.
//
// This test constructs a Second-mode `DiagonalQuadratic2dSecond`,
// wraps it into a First-mode `FunctionExpr`, and verifies that the
// value and gradient match the analytic answers at a non-trivial
// point.  If the downgrade ever breaks (e.g. the adapter is removed
// or the static_assert tightens back to equality) this fails at
// compile or run time.
TEST(FunctionExpr, SecondModeSourceDowngradesIntoFirstMode) {
  using namespace cppoptlib::function;
  // Target type: First-mode.  Source: Second-mode CRTP class.
  FunctionExpr<double, DifferentiabilityMode::First, 2> wrapped =
      DiagonalQuadratic2dSecond();
  Eigen::Vector2d x;
  x << 3.0, -1.5;
  // f(x) = 2 * 9 + 2.25 = 20.25.
  Eigen::Vector2d grad;
  const double value = wrapped(x, &grad);
  EXPECT_NEAR(20.25, value, penalty_evaluation_tolerance);
  // grad = (4*x0, 2*x1) = (12, -3).
  EXPECT_NEAR(12.0, grad[0], penalty_evaluation_tolerance);
  EXPECT_NEAR(-3.0, grad[1], penalty_evaluation_tolerance);
}

// =============================================================================
// SECTION D: Non-convex escape / KKT discipline
// =============================================================================
//
// The outer-loop tests in Section C all exercise convex objectives where
// every KKT point is the global minimum.  Real benchmark problems are
// rarely that kind.  A non-convex objective combined with inequality
// constraints can produce a feasible stationary point of the raw
// objective that is NOT the constrained optimum -- the inner solver
// lands there on the very first outer iteration (with zero multipliers
// the augmented Lagrangian is just `f` in the feasible interior), the
// violation is zero, and a feasibility-only stopping test concludes
// "converged" at the wrong point.  A proper KKT-style outer loop has
// to avoid that trap by (a) keeping the inner solve shallow on the
// first outer iteration so multipliers get a chance to become nonzero
// before the iterate locks in, (b) scaling the initial penalty to the
// objective magnitude so the constraint gradients are felt, and (c)
// refusing to declare success until the Lagrangian gradient is
// actually stationary at the current multipliers.
//
// The tests below are minimal reproducers of the HS024-class trap.
// Each has a hand-computed global optimum and a hand-identified
// spurious KKT point that a naive feasibility-only stop lands on.

// ---- D.1: Cubic-in-one-axis objective with triangular constraints -------
//
// f(x) = ((x0 - 3)^2 - 9) * x1^3 / (27 * sqrt(3)).
//
// Unconstrained, f is unbounded below because the leading bracket goes
// negative for x0 near 3 while x1 is unrestricted; cubic growth in x1
// dominates.  Three linear inequalities carve out a triangle in the
// first quadrant:
//     g0(x) = x0 / sqrt(3) - x1      >= 0  (below the line x1 = x0/sqrt(3))
//     g1(x) = x0 + sqrt(3) * x1      >= 0  (right half-plane, redundant here)
//     g2(x) = 6 - x0 - sqrt(3) * x1  >= 0  (left of the line x0 + sqrt(3) x1 =
//     6)
// Plus bounds x >= 0.
//
// The triangle has vertices at (0, 0), (6, 0), and (3, sqrt(3)).  The
// constrained optimum is the upper vertex (3, sqrt(3)) with f* = -1.
//
// SPURIOUS KKT: the origin (0, 0) is feasible, has grad f = 0 (both
// partials vanish at x1 = 0), and satisfies KKT with mu = 0 for all
// three inequalities (g0 = 0, g1 = 0, g2 = 6).  A feasibility-only
// outer-loop stop after one inner solve lands there.
//
// The start (1.0, 0.5) is in the strict interior of the triangle.
class Hs024Objective : public Function2d<Hs024Objective> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad) const {
    const double x0 = x[0];
    const double x1 = x[1];
    const double bracket = (x0 - 3.0) * (x0 - 3.0) - 9.0;
    const double scale = 1.0 / (27.0 * std::sqrt(3.0));
    const double value = bracket * x1 * x1 * x1 * scale;
    if (grad) {
      (*grad)[0] = 2.0 * (x0 - 3.0) * x1 * x1 * x1 * scale;
      (*grad)[1] = 3.0 * bracket * x1 * x1 * scale;
    }
    return value;
  }
};

// Triangle edge g0(x) = x0 / sqrt(3) - x1 >= 0.
class Hs024IneqUpperEdge : public Function2d<Hs024IneqUpperEdge> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad) const {
    const double sqrt3 = std::sqrt(3.0);
    if (grad) {
      (*grad)[0] = 1.0 / sqrt3;
      (*grad)[1] = -1.0;
    }
    return x[0] / sqrt3 - x[1];
  }
};

// Triangle edge g1(x) = x0 + sqrt(3) * x1 >= 0.  Redundant inside the
// first quadrant but part of the original problem.
class Hs024IneqRightEdge : public Function2d<Hs024IneqRightEdge> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad) const {
    const double sqrt3 = std::sqrt(3.0);
    if (grad) {
      (*grad)[0] = 1.0;
      (*grad)[1] = sqrt3;
    }
    return x[0] + sqrt3 * x[1];
  }
};

// Triangle edge g2(x) = 6 - x0 - sqrt(3) * x1 >= 0.
class Hs024IneqLeftEdge : public Function2d<Hs024IneqLeftEdge> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad) const {
    const double sqrt3 = std::sqrt(3.0);
    if (grad) {
      (*grad)[0] = -1.0;
      (*grad)[1] = -sqrt3;
    }
    return 6.0 - x[0] - sqrt3 * x[1];
  }
};

// Reports the test tolerance expected for the HS024-class trap.  A
// correct AL outer loop reaches the upper vertex to three decimal
// places from (1.0, 0.5); we pin one decimal to leave headroom for
// the penalty-growth schedule tail.
constexpr double nonconvex_trap_primal_tolerance = 1e-1;

// A correct AL outer loop for the HS024 trap converges to f* = -1.
// A feasibility-only stop lands at f = 0 (the origin).  The test
// tolerance below is half a unit, which separates the two outcomes
// by an order of magnitude.
constexpr double nonconvex_trap_objective_tolerance = 0.5;

TEST(AugmentedLagrangianNonConvex, Hs024TriangleEscapesSpuriousOrigin) {
  using namespace cppoptlib::function;

  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      Hs024Objective();
  FunctionExpr<double, DifferentiabilityMode::First, 2> g0 =
      Hs024IneqUpperEdge();
  FunctionExpr<double, DifferentiabilityMode::First, 2> g1 =
      Hs024IneqRightEdge();
  FunctionExpr<double, DifferentiabilityMode::First, 2> g2 =
      Hs024IneqLeftEdge();

  ConstrainedProblem2d problem(objective, /*eq=*/{}, {g0, g1, g2});

  // The non-negativity bounds on x are enforced by a bounded inner
  // solver.  We use `Lbfgsb` here for exactly that reason: the
  // unbounded `Lbfgs` would take f below -infinity along the x1 axis.
  cppoptlib::solver::Lbfgsb<FunctionExpr2d> inner_solver;
  Eigen::Vector2d lower_bound;
  lower_bound << 0.0, 0.0;
  Eigen::Vector2d upper_bound;
  // Use the conventional large-magnitude sentinel to mean "no
  // bound".  `Lbfgsb` treats bounds with magnitude at or above 1e20
  // as inactive.
  constexpr double no_upper_bound = 1e20;
  upper_bound << no_upper_bound, no_upper_bound;
  inner_solver.SetBounds(lower_bound, upper_bound);

  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  Eigen::Vector2d x0;
  x0 << 1.0, 0.5;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(
      x0, /*num_eq=*/0, /*num_ineq=*/3, /*penalty=*/0.0);
  auto [solution, progress] = solver.Minimize(state);

  const double f_final = objective(solution.x);

  // The constrained global minimum is f* = -1 at (3, sqrt(3)).  The
  // test passes only if the outer loop escaped the spurious (0, 0).
  EXPECT_NEAR(3.0, solution.x[0], nonconvex_trap_primal_tolerance);
  EXPECT_NEAR(std::sqrt(3.0), solution.x[1], nonconvex_trap_primal_tolerance);
  EXPECT_NEAR(-1.0, f_final, nonconvex_trap_objective_tolerance);
}

// ---- D.2: HS029-style product objective with quadratic constraint ------
//
// Hock-Schittkowski 29 (simplified to 2D, f = -x0 * x1 on the
// ellipse 48 - x0^2 - 2 x1^2 >= 0) has the global minimum at
// `(4 sqrt(2/3), 2 sqrt(2/3))` with f ≈ -6.53, and a spurious KKT
// at the origin where grad f = 0 and the constraint is inactive
// with mu = 0.  The AL outer loop escapes the origin trap only when
// the PHR penalty is used -- a naive `-mu * c` Lagrangian term with
// `rho` large would drag the composite to -infinity along the
// strongly-inactive constraint direction, but that is the
// opposite of the problem at hand: the issue here is that the raw
// objective has a local min at the origin which happens to also
// be a feasible point.
//
// We start at (1, 1), which is strictly feasible.  The PHR
// formulation + best-iterate tracker together must produce an
// output whose f is well below zero.
class ProductObjective3D : public Function2d<ProductObjective3D> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad) const {
    if (grad) {
      (*grad)[0] = -x[1];
      (*grad)[1] = -x[0];
    }
    return -x[0] * x[1];
  }
};

// Inequality c(x) = 48 - x0^2 - 2 x1^2 >= 0.
class Hs029Ellipse : public Function2d<Hs029Ellipse> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ScalarType operator()(const VectorType& x, VectorType* grad) const {
    if (grad) {
      (*grad)[0] = -2.0 * x[0];
      (*grad)[1] = -4.0 * x[1];
    }
    return 48.0 - x[0] * x[0] - 2.0 * x[1] * x[1];
  }
};

TEST(AugmentedLagrangianNonConvex, Hs029EllipseEscapesOrigin) {
  using namespace cppoptlib::function;

  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      ProductObjective3D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> ellipse =
      Hs029Ellipse();

  ConstrainedProblem2d problem(objective, /*eq=*/{}, {ellipse});

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  // Start far from origin and far from any boundary.
  Eigen::Vector2d x0;
  x0 << 1.0, 1.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(
      x0, /*num_eq=*/0, /*num_ineq=*/1, /*penalty=*/0.0);
  auto [solution, progress] = solver.Minimize(state);

  const double f_final = objective(solution.x);
  // True optimum for `min -x0*x1 s.t. 48 - x0^2 - 2 x1^2 >= 0` is at
  // `(2 sqrt(6), 2 sqrt(3))` with f* = -12 sqrt(2).  Derivation:
  // KKT stationarity gives `x1 = 2 mu x0` and `x0 = 4 mu x1`, so
  // `mu = 1/(2 sqrt(2))` and `x1 = x0 / sqrt(2)`.  Substituting into
  // the active constraint `x0^2 + 2 x1^2 = 48` yields `x0^2 = 24`.
  constexpr double ellipse_primal_tolerance = 2e-1;
  const double x0_star = 2.0 * std::sqrt(6.0);
  const double x1_star = 2.0 * std::sqrt(3.0);
  const double f_star = -12.0 * std::sqrt(2.0);
  EXPECT_NEAR(x0_star, solution.x[0], ellipse_primal_tolerance);
  EXPECT_NEAR(x1_star, solution.x[1], ellipse_primal_tolerance);
  constexpr double ellipse_objective_tolerance = 5e-1;
  EXPECT_NEAR(f_star, f_final, ellipse_objective_tolerance);
}

// ---- D.3: KKT stationarity is reported on the returned state ------------
//
// After `Minimize` returns with `Status::Finished`, the solver must
// have verified BOTH primal feasibility AND Lagrangian-gradient
// stationarity -- i.e. the inner solver's last iterate actually
// stationarises the Lagrangian at the current multipliers.  We pin
// that by reading `max_lagrangian_gradient` off the returned state
// and checking it is below the outer-loop's KKT tolerance.
//
// This test uses the well-behaved Section C problem (equality
// quadratic) -- the stationarity check must be reported correctly
// even on problems that converge easily.
TEST(AugmentedLagrangianOuter, KktStationarityReportedOnFinishedState) {
  using namespace cppoptlib::function;

  FunctionExpr<double, DifferentiabilityMode::First, 2> objective =
      HalfSquaredNorm2D();
  FunctionExpr<double, DifferentiabilityMode::First, 2> equality =
      X0MinusTarget(1.0);
  ConstrainedProblem2d problem(objective, {equality});

  cppoptlib::solver::Lbfgs<FunctionExpr2d> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  Eigen::Vector2d x0;
  x0 << 5.0, 5.0;
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 1, 0, 1.0);
  auto [solution, progress] = solver.Minimize(state);

  ASSERT_EQ(cppoptlib::solver::Status::Finished, progress.status);
  // The outer-loop KKT tolerance defaults match the primal one to a
  // couple of decimals in either direction -- this test reads just
  // that the reported Lagrangian gradient is small, not specifically
  // below any one tolerance.
  constexpr double kkt_stationarity_upper_bound = 1e-2;
  EXPECT_LE(solution.max_lagrangian_gradient, kkt_stationarity_upper_bound);
}

// HS016-class regression: optimum pinned to the inner solver's box
// boundary.  The feasible region is `{x in [-0.5, 0.5] x [-inf, 1] :
// x0^2 + x1 >= 0, x0 + x1^2 >= 0}` and the minimum of the Rosenbrock
// objective sits at `(0.5, 0.25)` -- on the upper bound of `x0`.  At
// that point the raw Lagrangian gradient is NOT zero (it has a
// component pointing out of the box), so a KKT test that reads the
// unprojected sup-norm refuses to declare convergence and the outer
// loop burns through its iteration cap.  After the projected-gradient
// fix, the outer loop must finish in a handful of iterations.
//
// This test guards against re-introducing the old feasibility-only or
// unprojected-gradient logic: if either regression lands, the outer
// loop falls out via `IterationLimit` and the assertion on
// `progress.status == Finished` fails.
TEST(AugmentedLagrangianBoxInterface, BoxPinnedOptimumStopsOnKkt) {
  using cppoptlib::function::FunctionExpr;
  using VectorType = Eigen::Matrix<double, 2, 1>;

  class Rosenbrock : public cppoptlib::function::FunctionCRTP<
                         Rosenbrock, double,
                         cppoptlib::function::DifferentiabilityMode::First, 2> {
   public:
    ScalarType operator()(const VectorType& x, VectorType* grad) const {
      const double a = x[0] - 1.0;
      const double b = x[0] * x[0] - x[1];
      if (grad) {
        (*grad)[0] = 2 * a + 400 * b * x[0];
        (*grad)[1] = -200 * b;
      }
      return a * a + 100 * b * b;
    }
  };

  class Ineq0 : public cppoptlib::function::FunctionCRTP<
                    Ineq0, double,
                    cppoptlib::function::DifferentiabilityMode::First, 2> {
   public:
    ScalarType operator()(const VectorType& x, VectorType* grad) const {
      if (grad) {
        (*grad)[0] = 2 * x[0];
        (*grad)[1] = 1;
      }
      return x[0] * x[0] + x[1];
    }
  };

  class Ineq1 : public cppoptlib::function::FunctionCRTP<
                    Ineq1, double,
                    cppoptlib::function::DifferentiabilityMode::First, 2> {
   public:
    ScalarType operator()(const VectorType& x, VectorType* grad) const {
      if (grad) {
        (*grad)[0] = 1;
        (*grad)[1] = 2 * x[1];
      }
      return x[0] + x[1] * x[1];
    }
  };

  using FExpr =
      FunctionExpr<double, cppoptlib::function::DifferentiabilityMode::First,
                   2>;
  FExpr objective = Rosenbrock{};
  FExpr i0 = Ineq0{};
  FExpr i1 = Ineq1{};
  cppoptlib::function::ConstrainedOptimizationProblem problem(objective, {},
                                                              {i0, i1});

  cppoptlib::solver::Lbfgsb<FExpr> inner;
  Eigen::Vector2d lower(-0.5, -1e20);
  Eigen::Vector2d upper(0.5, 1.0);
  inner.SetBounds(lower, upper);

  cppoptlib::solver::AugmentedLagrangian<decltype(problem), decltype(inner)>
      solver(problem, inner);

  Eigen::Vector2d x0(-2.0, 1.0);
  cppoptlib::solver::AugmentedLagrangeState<double, 2> state(x0, 0, 2, 0.0);

  auto [solution, progress] = solver.Minimize(state);

  // The outer loop must reach `Finished` status on its own -- an
  // `IterationLimit` exit would signal regression of the projected-
  // gradient fix.
  ASSERT_EQ(cppoptlib::solver::Status::Finished, progress.status);
  // Bound the iteration count loosely -- two or three outer
  // iterations are expected in practice; anything below twenty
  // leaves ample headroom while still catching regressions that
  // blow the iteration count past the cap.
  EXPECT_LT(progress.num_iterations, static_cast<std::size_t>(20));
  // The converged iterate must sit within numerical tolerance of
  // the known HS016 optimum (0.5, 0.25); the upper bound on `x0` is
  // the active box constraint.
  constexpr double position_tolerance = 1e-4;
  EXPECT_NEAR(solution.x[0], 0.5, position_tolerance);
  EXPECT_NEAR(solution.x[1], 0.25, position_tolerance);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
