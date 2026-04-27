// Copyright 2026, https://github.com/PatWie/CppNumericalSolvers
//
// Unit tests for the Hager-Zhang line search.  Each test drives the search
// on an analytic 1-D function so the expected outcome is computable in
// closed form.  We embed each 1-D test function as a one-dimensional Eigen
// vector and use `search_direction = 1` so that `alpha` directly indexes
// `phi(alpha) = f(alpha)`.
#undef NDEBUG

#include "cppoptlib/linesearch/hager_zhang.h"

#include <cmath>

#include "cppoptlib/function.h"
#include "gtest/gtest.h"

namespace {

// CRTP scalar function: `f(x) = a*x^2 + b*x + c` with `x` a 1-vector.
// The quadratic case 1 sets `a=1, b=-2, c=0` giving the classical
// `phi(a) = a^2 - 2a`, minimum at `x = 1`.
class Quadratic : public cppoptlib::function::FunctionCRTP<
                      Quadratic, double,
                      cppoptlib::function::DifferentiabilityMode::First, 1> {
 public:
  double a_coef;
  double b_coef;
  double c_coef;
  Quadratic(double a, double b, double c) : a_coef(a), b_coef(b), c_coef(c) {}
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    const double v = x[0];
    if (grad) {
      grad->resize(1);
      (*grad)[0] = 2 * a_coef * v + b_coef;
    }
    return a_coef * v * v + b_coef * v + c_coef;
  }
};

// `f(x) = x^3 - 3x + 2`, with `f'(0) = -3`.  Local minimum at `x = 1`
// where `f(1) = 0` and `f'(1) = 0`.
class Cubic
    : public cppoptlib::function::FunctionCRTP<
          Cubic, double, cppoptlib::function::DifferentiabilityMode::First, 1> {
 public:
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    const double v = x[0];
    if (grad) {
      grad->resize(1);
      (*grad)[0] = 3 * v * v - 3;
    }
    return v * v * v - 3 * v + 2;
  }
};

// `f(x) = 1e-8 * x + x^4` -- nearly flat near 0, a tiny negative slope
// `-1e-8` at the origin, quartic growth away from 0.  The HZ curvature
// condition `|dphi(c)| <= sigma * |dphi(0)|` is easy to satisfy, but the
// search must not stall iterating to max.
class FlatQuartic : public cppoptlib::function::FunctionCRTP<
                        FlatQuartic, double,
                        cppoptlib::function::DifferentiabilityMode::First, 1> {
 public:
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    const double v = x[0];
    if (grad) {
      grad->resize(1);
      (*grad)[0] = 1e-8 + 4 * v * v * v;
    }
    return 1e-8 * v + v * v * v * v;
  }
};

using Line1D = Eigen::Matrix<double, 1, 1>;

// Thin helper: run the Hager-Zhang search starting at `x0` with a unit
// descent direction and return `(alpha, f_at_alpha)`.
template <typename FunctionT>
std::pair<double, double> RunSearch(const FunctionT& function, double x0,
                                    double alpha_init) {
  Line1D x;
  x[0] = x0;
  Line1D s;
  s[0] = 1.0;
  Line1D x_out;
  double f_out = 0.0;
  Line1D g_out;
  Line1D g0;
  const double f0 = function(x, &g0);
  const double alpha =
      cppoptlib::solver::linesearch::HagerZhang<FunctionT, 1>::Search(
          x, f0, g0, s, function, alpha_init, &x_out, &f_out, &g_out);
  return {alpha, f_out};
}

}  // namespace

// ---- Case 1: convex quadratic `phi(a) = a^2 - 2a` ------------------------
// Starting at `a = 0` with `phi'(0) = -2`, the exact minimizer is `a = 1`.
// HZ should snap directly onto it (the first secant between `a=0` and
// `a=alpha_init` lands on 1 when alpha_init >= 1).
TEST(HagerZhangCase1, ConvexQuadraticMinimum) {
  Quadratic phi(1.0, -2.0, 0.0);
  auto [alpha, f_at] = RunSearch(phi, 0.0, 1.0);
  ASSERT_NEAR(1.0, alpha, 1e-6);
  ASSERT_NEAR(-1.0, f_at, 1e-6);
}

// ---- Case 2: cubic with a local minimum at a = 1 --------------------------
// `phi(a) = a^3 - 3a + 2`, `phi(0) = 2`, `phi'(0) = -3`.  The Wolfe curvature
// condition holds at `a = 1` where `phi'(1) = 0`.
TEST(HagerZhangCase2, CubicLocalMinimum) {
  Cubic phi;
  auto [alpha, f_at] = RunSearch(phi, 0.0, 1.0);
  ASSERT_NEAR(1.0, alpha, 1e-6);
  ASSERT_NEAR(0.0, f_at, 1e-6);
}

// ---- Case 3: strongly-ill-scaled quadratic --------------------------------
// `phi(a) = 1e6 * (a - 0.5)^2`, minimum at `a = 0.5`.  At `a = 0`,
// `phi(0) = 2.5e5`, `phi'(0) = -1e6`.  HZ should find the minimum without
// overshooting by more than a factor of two before recovering.
TEST(HagerZhangCase3, IllScaledQuadraticStaysBounded) {
  // `phi(a) = 1e6 * (a - 0.5)^2 = 1e6 a^2 - 1e6 a + 2.5e5`.
  Quadratic phi(1e6, -1e6, 2.5e5);
  auto [alpha, f_at] = RunSearch(phi, 0.0, 1.0);
  ASSERT_NEAR(0.5, alpha, 1e-6);
  ASSERT_NEAR(0.0, f_at, 1e-3);
  // Guard against wild overshoot in the returned step.
  ASSERT_GT(alpha, 0.0);
  ASSERT_LT(alpha, 1.0);
}

// ---- Case 4: very flat region ---------------------------------------------
// `phi(a) = 1e-8 * a + a^4`.  HZ must terminate on the curvature condition
// -- the strong-Wolfe curvature test `|dphi| <= sigma * |dphi_0|` with
// `dphi_0 = -1e-8` is trivially satisfied once a is small; but because the
// initial step a=1 overshoots to `phi(1) = 1 + 1e-8 > phi(0) = 0` the
// approximate-Wolfe branch must accept a bracketed interior point.  The
// algorithm must not run out of iterations.
TEST(HagerZhangCase4, FlatRegionTerminatesOnCurvature) {
  FlatQuartic phi;
  auto [alpha, f_at] = RunSearch(phi, 0.0, 1.0);
  // The step must be finite and positive.
  ASSERT_GT(alpha, 0.0);
  ASSERT_TRUE(std::isfinite(alpha));
  // Must have reduced the objective below the starting value `phi(0) = 0`.
  // With `phi(a) = 1e-8 a + a^4`, the true minimum is at
  // `a* = (1e-8 / 4)^{1/3} ~= 1.357e-3`, where `phi(a*) ~= -1.017e-11`.
  ASSERT_LE(f_at, 0.0);
}
