// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#undef NDEBUG

#include <functional>
#include <iostream>
#include <limits>
#include <list>

#include "gtest/gtest.h"
#include "include/cppoptlib/solver/bfgs.h"
#include "include/cppoptlib/solver/conjugated_gradient_descent.h"
#include "include/cppoptlib/solver/gradient_descent.h"
#include "include/cppoptlib/solver/lbfgs.h"
#include "include/cppoptlib/solver/lbfgsb.h"
#include "include/cppoptlib/solver/newton_descent.h"

constexpr double PRECISION = 1e-4;
constexpr auto FUNCTION_DIM = 2;

template <typename scalar_t>
class RosenbrockValue : public cppoptlib::function::Function<scalar_t> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using vector_t = typename cppoptlib::function::Function<scalar_t>::vector_t;

  scalar_t operator()(const vector_t &x) const override {
    const scalar_t t1 = (1 - x[0]);
    const scalar_t t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  }
};

template <typename scalar_t>
class RosenbrockGradient : public cppoptlib::function::Function<scalar_t> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using vector_t = typename cppoptlib::function::Function<scalar_t>::vector_t;

  scalar_t operator()(const vector_t &x) const override {
    const scalar_t t1 = (1 - x[0]);
    const scalar_t t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  }

  void Gradient(const vector_t &x, vector_t *grad) const override {
    (*grad)[0] = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
    (*grad)[1] = 200 * (x[1] - x[0] * x[0]);
  }
};

template <typename scalar_t>
class RosenbrockFull : public cppoptlib::function::Function<scalar_t> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using vector_t = typename cppoptlib::function::Function<scalar_t>::vector_t;
  using hessian_t = typename cppoptlib::function::Function<scalar_t>::hessian_t;

  scalar_t operator()(const vector_t &x) const override {
    const scalar_t t1 = (1 - x[0]);
    const scalar_t t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  }

  void Gradient(const vector_t &x, vector_t *grad) const override {
    (*grad)[0] = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
    (*grad)[1] = 200 * (x[1] - x[0] * x[0]);
  }

  void Hessian(const vector_t &x, hessian_t *hessian) const override {
    (*hessian)(0, 0) = 1200 * x[0] * x[0] - 400 * x[1] + 1;
    (*hessian)(0, 1) = -400 * x[0];
    (*hessian)(1, 0) = -400 * x[0];
    (*hessian)(1, 1) = 200;
  }
};

template <class T>
class GradientDescentTest : public testing::Test {};
template <class T>
class ConjugatedGradientDescentTest : public testing::Test {};
template <class T>
class NewtonDescentTest : public testing::Test {};
template <class T>
class BfgsTest : public testing::Test {};
template <class T>
class LbfgsTest : public testing::Test {};
template <class T>
class LbfgsbTest : public testing::Test {};

#define SOLVE_PROBLEM(sol, func, a, b, fx)                                \
  using Function = func<TypeParam>;                                       \
  using Solver = sol<Function>;                                           \
  Function f;                                                             \
  typename Function::vector_t x(FUNCTION_DIM);                            \
  x << a, b;                                                              \
  Solver solver;                                                          \
  solver.SetStepCallback(                                                 \
      cppoptlib::solver::GetEmptyStepCallback<                            \
          typename Function::scalar_t, typename Function::vector_t,       \
          typename Function::hessian_t>());                               \
  auto [solution, solver_state] = solver.Minimize(f, x);                  \
  if (solver_state.status == cppoptlib::solver::Status::IterationLimit) { \
    std::cout << solver_state.status << std::endl;                        \
  }                                                                       \
  EXPECT_NEAR(fx, f(solution.x), PRECISION);

#define LBFGSB_SOLVE_PROBLEM(sol, func, a, b, fx)                         \
  using Function = func<TypeParam>;                                       \
  using Solver = sol<Function>;                                           \
  Function f;                                                             \
  typename Function::vector_t x(FUNCTION_DIM);                            \
  x << a, b;                                                              \
  Solver solver{                                                          \
      Function::vector_t::Constant(                                       \
          FUNCTION_DIM,                                                   \
          std::numeric_limits<typename Function::scalar_t>::lowest()),    \
      Function::vector_t::Constant(                                       \
          FUNCTION_DIM,                                                   \
          std::numeric_limits<typename Function::scalar_t>::max())};      \
  solver.SetStepCallback(                                                 \
      cppoptlib::solver::GetEmptyStepCallback<                            \
          typename Function::scalar_t, typename Function::vector_t,       \
          typename Function::hessian_t>());                               \
  auto [solution, solver_state] = solver.Minimize(f, x);                  \
  if (solver_state.status == cppoptlib::solver::Status::IterationLimit) { \
    std::cout << solver_state.status << std::endl;                        \
  }                                                                       \
  EXPECT_NEAR(fx, f(solution.x), PRECISION);

typedef ::testing::Types<double> DoublePrecision;
TYPED_TEST_CASE(GradientDescentTest, DoublePrecision);
TYPED_TEST_CASE(ConjugatedGradientDescentTest, DoublePrecision);
TYPED_TEST_CASE(NewtonDescentTest, DoublePrecision);
TYPED_TEST_CASE(BfgsTest, DoublePrecision);
TYPED_TEST_CASE(LbfgsTest, DoublePrecision);
TYPED_TEST_CASE(LbfgsbTest, DoublePrecision);

#define SOLVER_SETUP(sol, func)                                               \
  TYPED_TEST(sol##Test, func##Far){SOLVE_PROBLEM(                             \
      cppoptlib::solver::sol, func, 15.0, 8.0, 0.0)} TYPED_TEST(sol##Test,    \
                                                                func##Near) { \
    SOLVE_PROBLEM(cppoptlib::solver::sol, func, -1.0, 2.0, 0.0)               \
  }

#define LBFGSB_SOLVER_SETUP(sol, func)                                        \
  TYPED_TEST(sol##Test, func##Far){LBFGSB_SOLVE_PROBLEM(                      \
      cppoptlib::solver::sol, func, 15.0, 8.0, 0.0)} TYPED_TEST(sol##Test,    \
                                                                func##Near) { \
    LBFGSB_SOLVE_PROBLEM(cppoptlib::solver::sol, func, -1.0, 2.0, 0.0)        \
  }

SOLVER_SETUP(GradientDescent, RosenbrockValue)
SOLVER_SETUP(GradientDescent, RosenbrockGradient)
SOLVER_SETUP(ConjugatedGradientDescent, RosenbrockValue)
SOLVER_SETUP(ConjugatedGradientDescent, RosenbrockGradient)
SOLVER_SETUP(Bfgs, RosenbrockValue)
SOLVER_SETUP(Bfgs, RosenbrockGradient)
SOLVER_SETUP(Lbfgs, RosenbrockValue)
SOLVER_SETUP(Lbfgs, RosenbrockGradient)
LBFGSB_SOLVER_SETUP(Lbfgsb, RosenbrockValue)
LBFGSB_SOLVER_SETUP(Lbfgsb, RosenbrockGradient)
SOLVER_SETUP(NewtonDescent, RosenbrockFull)

// simple function y <- 3*a-b
template <class scalar_t>
class SimpleFunction : public cppoptlib::function::Function<scalar_t> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using vector_t = typename cppoptlib::function::Function<scalar_t>::vector_t;
  using hessian_t = typename cppoptlib::function::Function<scalar_t>::hessian_t;

  scalar_t operator()(const vector_t &x) const override {
    return 3 * x[0] * x[0] - x[1] * x[0];
  }
};

template <class T>
class CentralDifferenceState : public testing::Test {};
TYPED_TEST_CASE(CentralDifferenceState, DoublePrecision);

TYPED_TEST(CentralDifferenceState, Gradient) {
  typename SimpleFunction<TypeParam>::vector_t x0(2);
  x0(0) = 0;
  x0(1) = 0;

  SimpleFunction<TypeParam> f;

  const auto state = f.Eval(x0);

  EXPECT_NEAR(state.gradient(0), 36 * x0(0) - x0(1), PRECISION);
  EXPECT_NEAR(state.gradient(1), -x0(0), PRECISION);
}

template <class T>
class CentralDifference : public testing::Test {};
TYPED_TEST_CASE(CentralDifference, DoublePrecision);

TYPED_TEST(CentralDifference, Gradient) {
  typename SimpleFunction<TypeParam>::vector_t x0(FUNCTION_DIM);
  typename SimpleFunction<TypeParam>::vector_t gradient(FUNCTION_DIM);
  x0(0) = 0;
  x0(1) = 0;

  SimpleFunction<TypeParam> f;

  // check from fast/bad to slower/better approximation of the gradient
  for (int accuracy = 0; accuracy < 4; ++accuracy) {
    cppoptlib::utils::ComputeFiniteGradient(f, x0, &gradient, accuracy);

    EXPECT_NEAR(gradient(0), 36 * x0(0) - x0(1), PRECISION);
    EXPECT_NEAR(gradient(1), -x0(0), PRECISION);
  }
}

TYPED_TEST(CentralDifference, Hessian) {
  typename SimpleFunction<TypeParam>::vector_t x0(FUNCTION_DIM);
  x0(0) = 0;
  x0(1) = 0;

  SimpleFunction<TypeParam> f;
  auto state = f.Eval(x0);

  EXPECT_NEAR((*state.hessian)(0, 0), 6, PRECISION);
  EXPECT_NEAR((*state.hessian)(1, 0), -1, PRECISION);
  EXPECT_NEAR((*state.hessian)(0, 1), -1, PRECISION);
  EXPECT_NEAR((*state.hessian)(1, 1), 0, PRECISION);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
