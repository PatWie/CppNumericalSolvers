// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#undef NDEBUG

#include <functional>
#include <iostream>
#include <limits>
#include <list>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/bfgs.h"
#include "cppoptlib/solver/conjugated_gradient_descent.h"
#include "cppoptlib/solver/gradient_descent.h"
#include "cppoptlib/solver/lbfgs.h"
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
#include "cppoptlib/solver/lbfgsb.h"
#endif
#include "cppoptlib/solver/nelder_mead.h"
#include "cppoptlib/solver/newton_descent.h"
#include "cppoptlib/utils/derivatives.h"
#include "gtest/gtest.h"

constexpr double PRECISION = 1e-4;

template <class T, class F>
using FunctionX2 = cppoptlib::function::FunctionCRTP<
    F, T, cppoptlib::function::DifferentiabilityMode::None>;
template <class T, class F>
using FunctionX2_dx = cppoptlib::function::FunctionCRTP<
    F, T, cppoptlib::function::DifferentiabilityMode::First>;
template <class T, class F>
using FunctionX2_dxx = cppoptlib::function::FunctionCRTP<
    F, T, cppoptlib::function::DifferentiabilityMode::Second>;

template <class T>
class RosenbrockValue : public FunctionX2<T, RosenbrockValue<T>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using typename FunctionX2<T, RosenbrockValue<T>>::ScalarType;
  using typename FunctionX2<T, RosenbrockValue<T>>::VectorType;

  ScalarType operator()(const VectorType &x) const {
    const T t1 = (1 - x[0]);
    const T t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  }
};

template <class T>
class RosenbrockGradient : public FunctionX2_dx<T, RosenbrockGradient<T>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using typename FunctionX2_dx<T, RosenbrockGradient<T>>::ScalarType;
  using typename FunctionX2_dx<T, RosenbrockGradient<T>>::VectorType;

  ScalarType operator()(const VectorType &x,
                        VectorType *gradient = nullptr) const {
    const T t1 = (1 - x[0]);
    const T t2 = (x[1] - x[0] * x[0]);
    if (gradient) {
      *gradient = VectorType::Zero(2);
      (*gradient)[0] =
          -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
      (*gradient)[1] = 200 * (x[1] - x[0] * x[0]);
    }
    return t1 * t1 + 100 * t2 * t2;
  }
};

template <class T>
class RosenbrockFull : public FunctionX2_dxx<T, RosenbrockFull<T>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using typename FunctionX2_dxx<T, RosenbrockFull<T>>::ScalarType;
  using typename FunctionX2_dxx<T, RosenbrockFull<T>>::VectorType;
  using typename FunctionX2_dxx<T, RosenbrockFull<T>>::MatrixType;

  ScalarType operator()(const VectorType &x, VectorType *gradient = nullptr,
                        MatrixType *hessian = nullptr) const {
    const T t1 = (1 - x[0]);
    const T t2 = (x[1] - x[0] * x[0]);
    if (gradient) {
      *gradient = VectorType::Zero(2);
      (*gradient)[0] =
          -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
      (*gradient)[1] = 200 * (x[1] - x[0] * x[0]);
    }
    if (hessian) {
      (*hessian) = MatrixType::Zero(2, 2);
      (*hessian)(0, 0) = 1200 * x[0] * x[0] - 400 * x[1] + 1;
      (*hessian)(0, 1) = -400 * x[0];
      (*hessian)(1, 0) = -400 * x[0];
      (*hessian)(1, 1) = 200;
    }
    return t1 * t1 + 100 * t2 * t2;
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
  typename Function::VectorType x(2);                                     \
  x << a, b;                                                              \
  auto initial_state = cppoptlib::function::FunctionState(x);             \
  Solver solver;                                                          \
  auto [solution, solver_state] = solver.Minimize(f, initial_state);      \
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
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
TYPED_TEST_CASE(LbfgsbTest, DoublePrecision);
#endif
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
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
SOLVER_SETUP(Lbfgsb, RosenbrockGradient)
#endif
SOLVER_SETUP(NewtonDescent, RosenbrockFull)
SOLVER_SETUP(NelderMead, RosenbrockValue)

// simple function y <- 3*a-b
template <class T>
class SimpleFunction : public FunctionX2<T, SimpleFunction<T>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using typename FunctionX2<T, SimpleFunction<T>>::ScalarType;
  using typename FunctionX2<T, SimpleFunction<T>>::VectorType;

  ScalarType operator()(const VectorType &x) const {
    return 3 * x[0] * x[0] - x[1] * x[0];
  }
};

template <class T>
class CentralDifference : public testing::Test {};
TYPED_TEST_CASE(CentralDifference, DoublePrecision);

TYPED_TEST(CentralDifference, Gradient) {
  typename SimpleFunction<TypeParam>::VectorType x0(2);
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
  typename SimpleFunction<TypeParam>::VectorType x0(2);
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
template <class F>
using Function2d = cppoptlib::function::FunctionCRTP<
    F, double, cppoptlib::function::DifferentiabilityMode::First, 2>;
using FunctionExpr2d1 = cppoptlib::function::FunctionExpr<
    double, cppoptlib::function::DifferentiabilityMode::First, 2>;

class SumObjective : public Function2d<SumObjective> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScalarType operator()(const VectorType &x,
                        VectorType *gradient = nullptr) const {
    if (gradient) {
      *gradient = VectorType::Ones(2);
    }
    return x.sum();
  }
};

class Circle : public Function2d<Circle> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Compute the squared norm.
  ScalarType operator()(const VectorType &x,
                        VectorType *gradient = nullptr) const {
    if (gradient) {
      *gradient = 2 * x;
    }
    return x.squaredNorm();
  }
};

template <class T>
class Constrained : public testing::Test {};
TYPED_TEST_CASE(Constrained, DoublePrecision);
TYPED_TEST(Constrained, Simple) {
  constexpr auto dim = 2;
  SumObjective::VectorType x(dim);
  x << 7, -4;
  // We have to start with one negative point. Otherwise, we would walk around
  // the boundary, which causes function values that might become bigger
  // (consider [0, 1] walking to [-1, -1] is almost impossible).

  // Define the objective function.
  cppoptlib::function::FunctionExpr objective = SumObjective();
  cppoptlib::function::FunctionExpr circle = Circle();

  cppoptlib::function::ConstrainedOptimizationProblem prob(
      objective,
      /* equality constraints */
      {cppoptlib::function::FunctionExpr(circle - 2)},
      /* inequality constraints */
      {cppoptlib::function::FunctionExpr(2 - circle)});
  cppoptlib::solver::Lbfgs<FunctionExpr2d1> inner_solver;
  cppoptlib::solver::AugmentedLagrangian<decltype(prob), decltype(inner_solver)>
      solver(inner_solver);
  cppoptlib::solver::AugmentedLagrangeState<double, 2> l_state(x, 1, 1, 1.0);

  // Run the solver.
  auto [solution, solver_state] = solver.Minimize(prob, l_state);
  EXPECT_NEAR(solution.x[0], -1, 1e-3);
  EXPECT_NEAR(solution.x[1], -1, 1e-3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
