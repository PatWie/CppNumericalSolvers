// Copyright 2026, https://github.com/PatWie/CppNumericalSolvers
//
// Unit tests for the `cstep` trial-interval update in the More-Thuente line
// search.  `cstep` is the safeguarded cubic/quadratic interpolation used to
// choose the next trial step on each line-search iteration.  We validate:
//   1. Structural invariants (info code, bracketing flag, clamp to
//   [tmin,tmax]).
//   2. Case-by-case correctness on scalar quadratic/cubic reference models
//      where the "right answer" is analytically computable.
//   3. The 0.66 safeguard that prevents the new trial from sitting too close
//      to one endpoint of a bracketed interval.
#undef NDEBUG

#include <cmath>
#include <limits>

#include "cppoptlib/function.h"
#include "cppoptlib/linesearch/more_thuente.h"
#include "gtest/gtest.h"

// A minimal first-order function instance just to satisfy the template
// parameter of `MoreThuente<FunctionType, Ord>`.  `cstep` itself is purely
// scalar and does not call `FunctionType::operator()`, but we still need a
// valid `FunctionType` to pull `ScalarType` / `VectorType` aliases from.
class ScalarFunctionStub
    : public cppoptlib::function::FunctionCRTP<
          ScalarFunctionStub, double,
          cppoptlib::function::DifferentiabilityMode::First> {
 public:
  ScalarType operator()(const VectorType&, VectorType* = nullptr) const {
    return 0.0;  // never called by cstep.
  }
};

using LineSearch =
    cppoptlib::solver::linesearch::MoreThuente<ScalarFunctionStub, /*Ord=*/1>;

// Convenience wrapper: call `cstep` with mutable local state and return the
// info code.  Mirrors the raw parameter order of the implementation so test
// setups read naturally.
static int CallCstep(double& stx, double& fx, double& dx, double& sty,
                     double& fy, double& dy, double& stp, double fp, double dp,
                     bool& brackt, double stpmin, double stpmax, int& info) {
  return LineSearch::cstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt,
                           stpmin, stpmax, info);
}

// ---- Case 1: fp > fx -----------------------------------------------------
// Reference model: phi(a) = 0.5 * a^2 - a, minimum at a* = 1.
//   phi(0) = 0,   phi'(0) = -1.
//   phi(3) = 1.5, phi'(3) = +2.
// Because phi is itself quadratic, both the cubic and quadratic interpolants
// in cstep must recover the exact minimizer a* = 1.
TEST(CstepCase1, QuadraticModelRecoversMinimizer) {
  double stx = 0.0, fx = 0.0, dx = -1.0;
  double sty = 0.0, fy = 0.0, dy = 0.0;
  double stp = 3.0, fp = 1.5, dp = 2.0;
  bool brackt = false;
  int info = 0;
  ASSERT_EQ(0, CallCstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt,
                         /*stpmin=*/0.0, /*stpmax=*/10.0, info));
  EXPECT_EQ(1, info);
  EXPECT_TRUE(brackt) << "Case 1 must set brackt to true.";
  EXPECT_NEAR(1.0, stp, 1e-12)
      << "For a purely quadratic model the interpolation should land on a*.";
  // After Case 1 the *unchanged* endpoint is `stx`, and the trial becomes
  // `sty`.  Verify the interval is oriented with the new trial on the right.
  EXPECT_EQ(0.0, stx);
  EXPECT_EQ(3.0, sty);
  EXPECT_EQ(1.5, fy);
  EXPECT_EQ(2.0, dy);
}

// ---- Case 2: fp <= fx and derivatives change sign -------------------------
// Reference model: phi(a) = 0.5 * (a - 2)^2, minimum at a* = 2.
//   phi(0) = 2,   phi'(0) = -2.
//   phi(3) = 0.5, phi'(3) = +1.
// fp < fx so we are NOT in Case 1; the derivatives flip sign (dx=-2, dp=+1),
// so Case 2 applies.  The secant through (stx, dx) and (stp, dp) zeros at
// a* = 2, so the quadratic minimizer must be 2.
TEST(CstepCase2, DerivativeSignFlipBracketsAndHitsMinimizer) {
  double stx = 0.0, fx = 2.0, dx = -2.0;
  double sty = 0.0, fy = 0.0, dy = 0.0;
  double stp = 3.0, fp = 0.5, dp = 1.0;
  bool brackt = false;
  int info = 0;
  ASSERT_EQ(0, CallCstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, 0.0,
                         10.0, info));
  EXPECT_EQ(2, info);
  EXPECT_TRUE(brackt) << "Case 2 must set brackt to true.";
  EXPECT_NEAR(2.0, stp, 1e-12)
      << "Secant root of a purely quadratic model must be a*.";
  // Both endpoints move: old stx kept on `sty`, trial became new `stx`.
  EXPECT_EQ(3.0, stx);
  EXPECT_EQ(0.5, fx);
  EXPECT_EQ(1.0, dx);
  EXPECT_EQ(0.0, sty);
  EXPECT_EQ(2.0, fy);
  EXPECT_EQ(-2.0, dy);
}

// ---- Case 3: same sign, |dp| < |dx|, not yet bracketed --------------------
// Reference model: phi(a) = 0.5 * (a - 4)^2, minimum at a* = 4.
//   phi(0) = 8,   phi'(0) = -4.
//   phi(1) = 4.5, phi'(1) = -3.
// fp < fx, derivatives have the same sign, |dp|=3 < |dx|=4: Case 3 applies.
// Not yet bracketed, so the algorithm should advance the trial step toward
// the minimizer (further from the current point among cubic/secant).
TEST(CstepCase3, NotBracketedAdvancesTowardMinimizer) {
  double stx = 0.0, fx = 8.0, dx = -4.0;
  double sty = 0.0, fy = 0.0, dy = 0.0;
  double stp = 1.0, fp = 4.5, dp = -3.0;
  bool brackt = false;
  int info = 0;
  constexpr double tmin = 0.0;
  constexpr double tmax = 20.0;
  ASSERT_EQ(0, CallCstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, tmin,
                         tmax, info));
  EXPECT_EQ(3, info);
  EXPECT_FALSE(brackt);  // still unbracketed because derivatives same-sign.
  EXPECT_GT(stp, 1.0) << "Should advance past the current trial toward a*=4.";
  EXPECT_LE(stp, tmax);
  // Case 3 adopts the trial as the new best: stx <- stp, fx <- fp, dx <- dp.
  EXPECT_EQ(1.0, stx);
  EXPECT_EQ(4.5, fx);
  EXPECT_EQ(-3.0, dx);
}

// ---- Case 4: same sign, |dp| >= |dx|, not yet bracketed -------------------
// We need stp > stx with derivative magnitude not decreasing.  Construct a
// model where the derivative stays flat or grows toward the trial.
//   phi(a) = 5 - a + 0.0 * a^2 (linear, phi' = -1 everywhere) is singular for
//   cubic interp, so use phi(a) = 5 - a - 0.01*a^3 which has growing |phi'|.
//   At a = 0: phi = 5,   phi' = -1.
//   At a = 1: phi = 3.99, phi' = -1 - 0.03 = -1.03.
// |dp| = 1.03 >= |dx| = 1, fp < fx, same-sign: Case 4.  Not bracketed, stp >
// stx, so the algorithm should extrapolate to stpmax.
TEST(CstepCase4, NotBracketedExtrapolatesToMax) {
  double stx = 0.0, fx = 5.0, dx = -1.0;
  double sty = 0.0, fy = 0.0, dy = 0.0;
  double stp = 1.0, fp = 3.99, dp = -1.03;
  bool brackt = false;
  int info = 0;
  constexpr double tmax = 50.0;
  ASSERT_EQ(0, CallCstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt,
                         /*stpmin=*/0.0, tmax, info));
  EXPECT_EQ(4, info);
  EXPECT_FALSE(brackt);
  EXPECT_EQ(tmax, stp)
      << "Unbracketed Case 4 with stp > stx must push to tmax.";
}

// ---- Result is always clamped to [stpmin, stpmax] -------------------------
// Drive Case 1 with an absurdly tight outer clamp and verify the safeguard.
TEST(CstepClamp, ResultAlwaysInsideStpminStpmax) {
  double stx = 0.0, fx = 0.0, dx = -1.0;
  double sty = 0.0, fy = 0.0, dy = 0.0;
  double stp = 3.0, fp = 1.5, dp = 2.0;
  bool brackt = false;
  int info = 0;
  // The unclamped answer for this Case 1 setup is 1.0; force clamp to 0.75.
  constexpr double stpmin = 0.1;
  constexpr double stpmax = 0.75;
  ASSERT_EQ(0, CallCstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin,
                         stpmax, info));
  EXPECT_GE(stp, stpmin);
  EXPECT_LE(stp, stpmax);
}

// ---- 0.66 safeguard when bracketed AND in a "bound" case ------------------
// After several cstep calls the interval shrinks.  If two consecutive bound
// cases (1 or 3) would leave the new trial near one endpoint, cstep rebases
// it to the 66% chord to avoid stagnation.  We construct an already-bracketed
// scenario where the raw cubic suggestion would land very close to `sty`.
TEST(CstepSafeguard, KeepsNewTrialInsideInnerTwoThirds) {
  // Interval [stx=0, sty=1] with dx<0 and dy>0 (bracketed); feed a new trial
  // stp=0.99 with fp > fx, which fires Case 1 (bound = true).  The 0.66 rule
  // must keep the result <= stx + 0.66 * (sty - stx) = 0.66.
  double stx = 0.0, fx = 0.0, dx = -1.0;
  double sty = 1.0, fy = 0.5, dy = 1.5;
  double stp = 0.99, fp = 0.49, dp = 1.4;
  bool brackt = true;
  int info = 0;
  ASSERT_EQ(0, CallCstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt,
                         /*stpmin=*/0.0, /*stpmax=*/2.0, info));
  EXPECT_EQ(1, info);
  EXPECT_TRUE(brackt);
  EXPECT_GE(stp, 0.0);
  EXPECT_LE(stp, 0.66 + 1e-12)
      << "0.66 safeguard must cap the new trial inside the inner two-thirds.";
}

// ---- Invariant: dx * (stp - stx) < 0 on entry is required -----------------
// The implementation reports an error (negative return) when this invariant
// is violated.  This guards against callers feeding non-descent triples.
TEST(CstepInvariants, RejectsNonDescentInput) {
  double stx = 0.0, fx = 0.0, dx = +1.0;  // wrong sign: dx * (stp - stx) > 0.
  double sty = 0.0, fy = 0.0, dy = 0.0;
  double stp = 1.0, fp = 0.5, dp = 0.5;
  bool brackt = false;
  int info = 0;
  EXPECT_EQ(-1, CallCstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, 0.0,
                          10.0, info));
}
