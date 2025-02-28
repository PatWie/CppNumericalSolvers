// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#undef NDEBUG

#include <functional>
#include <iostream>
#include <limits>
#include <list>

#include "gtest/gtest.h"
#include "include/cppoptlib/constrained_function.h"
#include "include/cppoptlib/function.h"
#include "include/cppoptlib/solver/augmented_lagrangian.h"
#include "include/cppoptlib/solver/bfgs.h"
#include "include/cppoptlib/solver/conjugated_gradient_descent.h"
#include "include/cppoptlib/solver/gradient_descent.h"
#include "include/cppoptlib/solver/lbfgs.h"
#include "include/cppoptlib/solver/lbfgsb.h"
#include "include/cppoptlib/solver/nelder_mead.h"
#include "include/cppoptlib/solver/newton_descent.h"
#include "include/cppoptlib/utils/derivatives.h"

constexpr double PRECISION = 1e-4;

template <class T>
using FunctionX2 =
    cppoptlib::function::Function<T, 2,
                                  cppoptlib::function::Differentiability::None>;
template <class T>
using FunctionX2_dx = cppoptlib::function::Function<
    T, 2, cppoptlib::function::Differentiability::First>;
template <class T>
using FunctionX2_dxx = cppoptlib::function::Function<
    T, 2, cppoptlib::function::Differentiability::Second>;

template <class T>
class RosenbrockValue : public FunctionX2<T> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using typename FunctionX2<T>::state_t;
  using typename FunctionX2<T>::scalar_t;
  using typename FunctionX2<T>::vector_t;

  state_t operator()(const vector_t &x) const override {
    state_t state;
    const T t1 = (1 - x[0]);
    const T t2 = (x[1] - x[0] * x[0]);
    state.value = t1 * t1 + 100 * t2 * t2;
    state.x = x;
    return state;
  }
};

template <class T>
class RosenbrockGradient : public FunctionX2_dx<T> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using typename FunctionX2_dx<T>::state_t;
  using typename FunctionX2_dx<T>::scalar_t;
  using typename FunctionX2_dx<T>::vector_t;

  state_t operator()(const vector_t &x) const override {
    state_t state;
    const T t1 = (1 - x[0]);
    const T t2 = (x[1] - x[0] * x[0]);
    state.value = t1 * t1 + 100 * t2 * t2;
    state.x = x;
    state.gradient = vector_t::Zero(2);
    state.gradient[0] =
        -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
    state.gradient[1] = 200 * (x[1] - x[0] * x[0]);
    return state;
  }
};

template <class T>
class RosenbrockFull : public FunctionX2_dxx<T> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using typename FunctionX2_dxx<T>::state_t;
  using typename FunctionX2_dxx<T>::scalar_t;
  using typename FunctionX2_dxx<T>::vector_t;
  using typename FunctionX2_dxx<T>::matrix_t;

  state_t operator()(const vector_t &x) const override {
    state_t state;
    const T t1 = (1 - x[0]);
    const T t2 = (x[1] - x[0] * x[0]);
    state.value = t1 * t1 + 100 * t2 * t2;
    state.x = x;
    state.gradient = vector_t::Zero(2);
    state.gradient[0] =
        -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
    state.gradient[1] = 200 * (x[1] - x[0] * x[0]);
    state.hessian = matrix_t::Zero(2, 2);
    state.hessian(0, 0) = 1200 * x[0] * x[0] - 400 * x[1] + 1;
    state.hessian(0, 1) = -400 * x[0];
    state.hessian(1, 0) = -400 * x[0];
    state.hessian(1, 1) = 200;
    return state;
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
template <class T>
class NelderMeadTest : public testing::Test {};

#define SOLVE_PROBLEM(sol, func, a, b, fx)                                \
  using Function = func<TypeParam>;                                       \
  using Solver = sol<Function>;                                           \
  Function f;                                                             \
  typename Function::vector_t x(2);                                       \
  x << a, b;                                                              \
  Solver solver;                                                          \
  auto [solution, solver_state] = solver.Minimize(f, f(x));               \
  if (solver_state.status == cppoptlib::solver::Status::IterationLimit) { \
    std::cout << solver_state.status << std::endl;                        \
  }                                                                       \
  EXPECT_NEAR(fx, solution.value, PRECISION);

typedef ::testing::Types<double> DoublePrecision;
TYPED_TEST_CASE(GradientDescentTest, DoublePrecision);
TYPED_TEST_CASE(ConjugatedGradientDescentTest, DoublePrecision);
TYPED_TEST_CASE(NewtonDescentTest, DoublePrecision);
TYPED_TEST_CASE(BfgsTest, DoublePrecision);
TYPED_TEST_CASE(LbfgsTest, DoublePrecision);
TYPED_TEST_CASE(LbfgsbTest, DoublePrecision);
TYPED_TEST_CASE(NelderMeadTest, DoublePrecision);

#define SOLVER_SETUP(sol, func)                                               \
  TYPED_TEST(sol##Test, func##Far){SOLVE_PROBLEM(                             \
      cppoptlib::solver::sol, func, 15.0, 8.0, 0.0)} TYPED_TEST(sol##Test,    \
                                                                func##Near) { \
    SOLVE_PROBLEM(cppoptlib::solver::sol, func, -1.0, 2.0, 0.0)               \
  }

SOLVER_SETUP(GradientDescent, RosenbrockGradient)
SOLVER_SETUP(ConjugatedGradientDescent, RosenbrockGradient)
SOLVER_SETUP(Bfgs, RosenbrockGradient)
SOLVER_SETUP(Lbfgs, RosenbrockGradient)
SOLVER_SETUP(Lbfgsb, RosenbrockGradient)
SOLVER_SETUP(NewtonDescent, RosenbrockFull)
// SOLVER_SETUP(NelderMead, RosenbrockValue)

// simple function y <- 3*a-b
template <class T>
class SimpleFunction : public FunctionX2<T> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using typename FunctionX2<T>::scalar_t;
  using typename FunctionX2<T>::vector_t;
  using typename FunctionX2<T>::state_t;

  state_t operator()(const vector_t &x) const override {
    state_t state;
    state.value = 3 * x[0] * x[0] - x[1] * x[0];
    return state;
  }
};

template <class T>
class CentralDifference : public testing::Test {};
TYPED_TEST_CASE(CentralDifference, DoublePrecision);

TYPED_TEST(CentralDifference, Gradient) {
  typename SimpleFunction<TypeParam>::vector_t x0(2);
  Eigen::Matrix<TypeParam, Eigen::Dynamic, 1> gradient =
      Eigen::Matrix<TypeParam, Eigen::Dynamic, 1>::Zero(2);
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
  typename SimpleFunction<TypeParam>::vector_t x0(2);
  x0(0) = 0;
  x0(1) = 0;

  SimpleFunction<TypeParam> f;

  Eigen::Matrix<TypeParam, Eigen::Dynamic, Eigen::Dynamic> hessian =
      Eigen::Matrix<TypeParam, Eigen::Dynamic, Eigen::Dynamic>::Zero(2, 2);
  cppoptlib::utils::ComputeFiniteHessian(f, x0, &hessian);

  EXPECT_NEAR(hessian(0, 0), 6, PRECISION);
  EXPECT_NEAR(hessian(1, 0), -1, PRECISION);
  EXPECT_NEAR(hessian(0, 1), -1, PRECISION);
  EXPECT_NEAR(hessian(1, 1), 0, PRECISION);
}

// constrained version
using Function2d = cppoptlib::function::Function<
    double, 2, cppoptlib::function::Differentiability::First>;
using Function2dC = cppoptlib::function::ConstrainedFunction<Function2d, 2>;

class Objective : public Function2d {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  state_t operator()(const vector_t &x) const override {
    state_t state;
    state.x = x;
    state.value = x.sum();
    state.gradient = vector_t::Ones(2);
    return state;
  }
};

class CircleBoundaryConstraint : public Function2d {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Enforces x[0]^2+x[1]^2 == 2 (on the circle boundary)
  state_t operator()(const vector_t &x) const override {
    state_t state;
    state.x = x;
    state.value = x.squaredNorm() - 2;
    state.gradient = 2 * x;
    return state;
  }
};

class InsideCircleConstraint : public Function2d {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Enforces x[0]^2+x[1]^2 <= 2 (inside the circle)
  state_t operator()(const vector_t &x) const override {
    state_t state;
    state.x = x;
    state.value = 2 - x.squaredNorm();
    state.gradient = -2 * x;
    return state;
  }
};

template <class T>
class Constrained : public testing::Test {};
TYPED_TEST_CASE(Constrained, DoublePrecision);
TYPED_TEST(Constrained, Simple) {
  using InnerSolver = cppoptlib::solver::Lbfgs<Function2d>;

  constexpr auto dim = 2;
  Objective::vector_t x(dim);
  x << 7, -4;
  // We have to start with one negative point. Otherwise, we would walk around
  // the boundary, which causes function values that might become bigger
  // (consider [0, 1] walking to [-1, -1] is almost impossible).

  Objective f;
  cppoptlib::function::NonNegativeConstraint<InsideCircleConstraint> c1;
  cppoptlib::function::ZeroConstraint<CircleBoundaryConstraint> c2;

  Function2dC L(&f, {&c1, &c2});
  cppoptlib::function::ConstrainedState<Function2dC> state =
      L(x, {0.0, 0.0}, 10.);

  cppoptlib::solver::AugmentedLagrangian<Function2dC, InnerSolver> solver;

  auto [solution, solver_state] = solver.Minimize(L, state);
  EXPECT_NEAR(solution.x[0], -1, 1e-3);
  EXPECT_NEAR(solution.x[1], -1, 1e-3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
