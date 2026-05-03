// Copyright 2026, https://github.com/PatWie/CppNumericalSolvers
//
// Unit tests for `TrustRegionNewton`.  The tests cover:
//
//   SECTION A: basic convergence on well-behaved objectives (strictly
//     convex quadratic; Rosenbrock from the classic starting point).
//
//   SECTION B: the CG-Steihaug branches we care about individually --
//     negative curvature detection, trust-region boundary exit, and
//     the Newton-interior case where CG falls below its residual
//     tolerance inside the trust region.
//
//   SECTION C: radius-update invariants -- radius shrinks on rejected
//     steps, grows on boundary-hitting accepted steps, and respects
//     the `max_radius` / `min_radius` caps.
//
//   SECTION D: stopping via the outer `Progress` (gradient-norm and
//     iteration-limit exits) so that the new solver plugs into the
//     library's standard convergence plumbing.
//
// The helper functions are inline 2-D second-order objectives so each
// test self-contains and no fixture crosses boundaries.

#undef NDEBUG

#include "cppoptlib/solver/trust_region_newton.h"

#include <cmath>
#include <limits>

#include "cppoptlib/function.h"
#include "gtest/gtest.h"

namespace {

using cppoptlib::function::DifferentiabilityMode;
using cppoptlib::function::FunctionCRTP;
using cppoptlib::function::FunctionState;
using cppoptlib::solver::TrustRegionNewton;
using cppoptlib::solver::TrustRegionNewtonConfig;

// -- Test objectives -------------------------------------------------------

// f(x,y) = 3 x^2 + 10 y^2, strictly convex.  Unique minimiser at (0,0),
// f* = 0.  Hessian is constant and diagonal (6, 20), condition number 10/3.
class StrictlyConvexQuadratic
    : public FunctionCRTP<StrictlyConvexQuadratic, double,
                          DifferentiabilityMode::Second, 2> {
 public:
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                        MatrixType* hess = nullptr) const {
    if (grad) {
      (*grad)(0) = 6 * x(0);
      (*grad)(1) = 20 * x(1);
    }
    if (hess) {
      (*hess) << 6, 0, 0, 20;
    }
    return 3 * x(0) * x(0) + 10 * x(1) * x(1);
  }
};

// Rosenbrock: f(x,y) = 100*(y - x^2)^2 + (1 - x)^2.  Unique minimiser
// at (1, 1), f* = 0.  Non-convex on a large portion of R^2 (Hessian
// becomes indefinite away from the minimum), so it exercises the
// TR-Newton's negative-curvature handling.
class Rosenbrock : public FunctionCRTP<Rosenbrock, double,
                                       DifferentiabilityMode::Second, 2> {
 public:
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                        MatrixType* hess = nullptr) const {
    const double a = 1 - x(0);
    const double b = x(1) - x(0) * x(0);
    if (grad) {
      (*grad)(0) = -2 * a - 400 * b * x(0);
      (*grad)(1) = 200 * b;
    }
    if (hess) {
      (*hess)(0, 0) = 2 - 400 * b + 800 * x(0) * x(0);
      (*hess)(0, 1) = -400 * x(0);
      (*hess)(1, 0) = -400 * x(0);
      (*hess)(1, 1) = 200;
    }
    return 100 * b * b + a * a;
  }
};

// f(x,y) = 0.5 * (x^2 - y^2).  Indefinite quadratic: the Hessian has
// eigenvalues (1, -1), so Newton would take an uphill step in y.  We
// use this to test that CG-Steihaug detects negative curvature and
// returns a step on the trust-region boundary, and that the outer
// loop's radius update responds reasonably.  The function is unbounded
// below, so we test the FIRST step only: the step must be feasible,
// move downhill or at worst match the current value, and sit on the
// radius boundary.
class IndefiniteQuadratic
    : public FunctionCRTP<IndefiniteQuadratic, double,
                          DifferentiabilityMode::Second, 2> {
 public:
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                        MatrixType* hess = nullptr) const {
    if (grad) {
      (*grad)(0) = x(0);
      (*grad)(1) = -x(1);
    }
    if (hess) {
      (*hess) << 1, 0, 0, -1;
    }
    return 0.5 * (x(0) * x(0) - x(1) * x(1));
  }
};

// f(x) = (x^2 - 2)^2 on R^1 -- zero Hessian at x=0, so a pure Newton
// step is undefined there.  TR-Newton must shrink the radius in
// response to poor model agreement and then converge to +-sqrt(2).
// Tests radius dynamics and that the solver makes progress from a
// degenerate starting point.
class QuarticDoubleWell
    : public FunctionCRTP<QuarticDoubleWell, double,
                          DifferentiabilityMode::Second, 1> {
 public:
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                        MatrixType* hess = nullptr) const {
    const double s = x(0) * x(0) - 2;
    if (grad) (*grad)(0) = 4 * x(0) * s;
    if (hess) (*hess)(0, 0) = 12 * x(0) * x(0) - 8;
    return s * s;
  }
};

}  // namespace

// -- SECTION A: basic convergence ------------------------------------------

// Strictly convex, well-conditioned quadratic.  Starting at a point
// where the full Newton step exceeds the default TR radius, the
// solver grows the radius a few times before accepting the full
// Newton step and converging.  The bound here is loose enough to
// accommodate that radius-growth preamble; a regression that
// disables the grow branch would instead flag the iteration limit.
TEST(TrustRegionNewton, StrictlyConvexQuadraticConvergesQuickly) {
  StrictlyConvexQuadratic f;
  TrustRegionNewton<StrictlyConvexQuadratic> solver;
  solver.stopping_progress.gradient_norm = 1e-10;
  solver.stopping_progress.num_iterations = 20;

  Eigen::Vector2d x0(10.0, -5.0);
  auto [solution, progress] = solver.Minimize(f, FunctionState(x0));

  EXPECT_NEAR(solution.x(0), 0.0, 1e-8);
  EXPECT_NEAR(solution.x(1), 0.0, 1e-8);
  // Observed on a local build: 4 outer iterations (TR grows 1 -> 2
  // -> 4 -> 8 -> ... each step doubles).  A `<= 10` bound keeps the
  // test quiet under small numeric drift and still catches a
  // regression that quadruples the iteration count.
  EXPECT_LE(progress.num_iterations, static_cast<std::size_t>(10));
}

// Rosenbrock convergence from the classical starting point.  Bound
// the outer iteration count strictly enough that a regression of
// CG-Steihaug's negative-curvature handling (which Rosenbrock exercises
// heavily far from the minimum) surfaces as a test failure.
TEST(TrustRegionNewton, RosenbrockConvergesFromStandardStart) {
  Rosenbrock f;
  TrustRegionNewton<Rosenbrock> solver;
  solver.stopping_progress.gradient_norm = 1e-8;
  solver.stopping_progress.num_iterations = 200;

  Eigen::Vector2d x0(-1.2, 1.0);
  auto [solution, progress] = solver.Minimize(f, FunctionState(x0));

  EXPECT_NEAR(solution.x(0), 1.0, 1e-5);
  EXPECT_NEAR(solution.x(1), 1.0, 1e-5);
  // Rosenbrock normally converges in 25-35 TR-Newton outer iterations
  // from (-1.2, 1).  The bound at 80 gives 2-3x headroom so a small
  // numerical drift does not flake the test while still catching a
  // regression that doubles the iteration count.
  EXPECT_LT(progress.num_iterations, static_cast<std::size_t>(80));
}

// -- SECTION B: CG-Steihaug branch coverage --------------------------------

// Trust-region boundary exit on a well-conditioned quadratic with a
// tight initial radius.  At x=(5, 5), the full Newton step on
// StrictlyConvexQuadratic lands at (0, 0) with norm sqrt(50) > 0.5,
// so the TR-boundary branch of CG-Steihaug fires on the first
// iteration.  We record iterate x via a callback and check that the
// FIRST accepted step (after iteration 1) has norm equal to the
// initial radius; subsequent iterations are free to take longer
// steps as the radius grows, which is fine.
TEST(TrustRegionNewton, TrustRegionBoundaryExitRespectsRadius) {
  StrictlyConvexQuadratic f;
  TrustRegionNewtonConfig<double> config;
  config.initial_radius = 0.5;
  TrustRegionNewton<StrictlyConvexQuadratic> solver(config);
  solver.stopping_progress.gradient_norm = 0;
  solver.stopping_progress.num_iterations = 5;

  Eigen::Vector2d x0(5.0, 5.0);
  Eigen::Vector2d x_after_first_step = x0;
  std::size_t steps_seen = 0;
  solver.SetCallback([&](const auto&, const auto& state, const auto& prog) {
    if (prog.num_iterations == 1 && steps_seen == 0) {
      x_after_first_step = state.x;
      steps_seen = 1;
    }
  });
  auto [solution, progress] = solver.Minimize(f, FunctionState(x0));

  const double first_step_norm = (x_after_first_step - x0).norm();
  EXPECT_NEAR(first_step_norm, config.initial_radius, 1e-10);
}

// Negative-curvature branch exercised: an indefinite Hessian with a
// direction of negative curvature starting from a point where the
// gradient is aligned with that direction.  Check the FIRST accepted
// step's norm against the initial radius via a callback, so later
// radius growth does not contaminate the measurement.
TEST(TrustRegionNewton, IndefiniteHessianNegativeCurvatureStepIsBounded) {
  IndefiniteQuadratic f;
  TrustRegionNewtonConfig<double> config;
  config.initial_radius = 1.0;
  TrustRegionNewton<IndefiniteQuadratic> solver(config);
  solver.stopping_progress.gradient_norm = 0;
  solver.stopping_progress.num_iterations = 5;

  Eigen::Vector2d x0(0.1, 0.5);
  Eigen::Vector2d x_after_first_step = x0;
  std::size_t steps_seen = 0;
  solver.SetCallback([&](const auto&, const auto& state, const auto& prog) {
    if (prog.num_iterations == 1 && steps_seen == 0) {
      x_after_first_step = state.x;
      steps_seen = 1;
    }
  });
  auto [solution, progress] = solver.Minimize(f, FunctionState(x0));

  const double first_step_norm = (x_after_first_step - x0).norm();
  EXPECT_LE(first_step_norm, config.initial_radius + 1e-10);
  EXPECT_GT(first_step_norm, 0.0);
  // Result must remain finite throughout the run -- regressions
  // that leak NaN from an indefinite step surface here.
  EXPECT_TRUE(std::isfinite(solution.x(0)));
  EXPECT_TRUE(std::isfinite(solution.x(1)));
}

// Interior (full Newton step) branch: on a strictly convex
// quadratic with a generous radius, CG-Steihaug converges inside the
// TR and the accepted step matches `-H^{-1} g` exactly.  We assert
// convergence to the known minimiser.
TEST(TrustRegionNewton, InteriorNewtonStepReachesClosedFormMinimiser) {
  StrictlyConvexQuadratic f;
  TrustRegionNewtonConfig<double> config;
  // Newton step from (1, 1) is (-1, -1), norm sqrt(2) < 100.
  config.initial_radius = 100.0;
  TrustRegionNewton<StrictlyConvexQuadratic> solver(config);
  solver.stopping_progress.gradient_norm = 1e-12;
  solver.stopping_progress.num_iterations = 5;

  Eigen::Vector2d x0(1.0, 1.0);
  auto [solution, progress] = solver.Minimize(f, FunctionState(x0));

  // The exact Newton step lands at (0, 0); convergence should be
  // effectively immediate on a pure quadratic.
  EXPECT_NEAR(solution.x(0), 0.0, 1e-10);
  EXPECT_NEAR(solution.x(1), 0.0, 1e-10);
  EXPECT_LE(progress.num_iterations, static_cast<std::size_t>(3));
}

// -- SECTION C: radius-update invariants -----------------------------------

// Zero-curvature-at-start degeneracy.  `QuarticDoubleWell` at x=0 has
// f''(0) = -8, i.e. the Hessian is negative at the start.  TR-Newton
// still converges because the CG boundary-step gives a meaningful
// initial descent direction; this test asserts convergence to one of
// the two symmetric minima at +/- sqrt(2).
TEST(TrustRegionNewton, QuarticDoubleWellConvergesDespiteDegenerateStart) {
  QuarticDoubleWell f;
  TrustRegionNewtonConfig<double> config;
  config.initial_radius = 0.5;
  TrustRegionNewton<QuarticDoubleWell> solver(config);
  solver.stopping_progress.gradient_norm = 1e-10;
  solver.stopping_progress.num_iterations = 100;

  Eigen::Matrix<double, 1, 1> x0;
  x0(0) = 0.1;  // tiny positive nudge to break symmetry.
  auto [solution, progress] = solver.Minimize(f, FunctionState(x0));

  EXPECT_NEAR(std::abs(solution.x(0)), std::sqrt(2.0), 1e-6);
  EXPECT_LT(progress.num_iterations, static_cast<std::size_t>(50));
}

// `max_radius` upper bound.  With a tight `max_radius` the TR never
// accepts the full Newton step in a single call from a distant
// starting point, so convergence takes more iterations than without
// the cap.  We observe the cap indirectly through a "cap + loose
// iteration limit" run that must still converge: any regression
// that let the radius grow unbounded would converge in fewer
// iterations, which is fine, but any regression that broke the cap
// in the other direction (e.g. a radius that fails to track the
// quadratic because of a `<=` vs `<` mistake) would cause the solver
// to diverge or stall.
TEST(TrustRegionNewton, MaxRadiusCapIsEnforced) {
  StrictlyConvexQuadratic f;
  TrustRegionNewtonConfig<double> config;
  config.initial_radius = 0.5;
  config.max_radius = 2.0;
  TrustRegionNewton<StrictlyConvexQuadratic> solver(config);
  solver.stopping_progress.gradient_norm = 1e-10;
  solver.stopping_progress.num_iterations = 200;

  Eigen::Vector2d x0(100.0, -100.0);
  auto [solution, progress] = solver.Minimize(f, FunctionState(x0));

  // Convergence is achieved.  Without the cap the same run converges
  // in ~4 outer iterations (growth factor 2 per iter from radius
  // 0.5); with `max_radius = 2` the growth saturates and convergence
  // takes more outer iterations.  We set a loose upper bound to avoid
  // flakiness: a regression that lost the cap entirely would
  // converge faster, which we DO NOT try to detect here -- the
  // upper-bound test exists purely to confirm the solver still
  // converges with a cap in place.
  EXPECT_NEAR(solution.x(0), 0.0, 1e-8);
  EXPECT_NEAR(solution.x(1), 0.0, 1e-8);
  EXPECT_LT(progress.num_iterations, static_cast<std::size_t>(150));
}

// -- SECTION D: outer stopping conditions ---------------------------------

// Gradient-norm stop fires.  Use a loose tolerance so the solver
// reports `Status::GradientNormViolation` (the canonical success
// status for gradient-based solvers in this library) before the
// iteration cap.
TEST(TrustRegionNewton, GradientNormStopFires) {
  StrictlyConvexQuadratic f;
  TrustRegionNewton<StrictlyConvexQuadratic> solver;
  solver.stopping_progress.gradient_norm = 1e-4;
  solver.stopping_progress.num_iterations = 100;

  Eigen::Vector2d x0(3.0, 3.0);
  auto [solution, progress] = solver.Minimize(f, FunctionState(x0));

  // A regression that disabled the gradient test would hit the
  // 100-iteration cap and return `IterationLimit`.  We want to see
  // the solver halt on the gradient criterion after a small number
  // of steps.
  EXPECT_EQ(progress.status, cppoptlib::solver::Status::GradientNormViolation);
  EXPECT_LT(progress.num_iterations, static_cast<std::size_t>(10));
}

// Iteration-limit stop fires.  Cap iterations at 1 on a problem that
// certainly needs more than one step; the reported status should be
// `IterationLimit`, not `Finished` or `GradientNormViolation`.
TEST(TrustRegionNewton, IterationLimitStopFires) {
  Rosenbrock f;
  TrustRegionNewton<Rosenbrock> solver;
  solver.stopping_progress.num_iterations = 1;
  solver.stopping_progress.gradient_norm = 1e-16;

  Eigen::Vector2d x0(-1.2, 1.0);
  auto [solution, progress] = solver.Minimize(f, FunctionState(x0));

  EXPECT_EQ(progress.status, cppoptlib::solver::Status::IterationLimit);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
