#include "gtest/gtest.h"
#include <iostream>
#include <functional>
#include <list>
#include "Meta.h"
#include "BfgsSolver.h"
#include "LbfgsSolver.h"
#include "LbfgsbSolver.h"
#include "GradientDescentSolver.h"
#include "ConjugateGradientSolver.h"
#include "NewtonDescentSolver.h"

#define PRECISION 1e-4

auto rosenbrock = [](const pwie::Vector &x) -> double {
  const double t1 = (1 - x[0]);
  const double t2 = (x[1] - x[0] * x[0]);
  return   t1 * t1 + 100 * t2 * t2;
};
auto Drosenbrock = [](const pwie::Vector x, pwie::Vector &grad) -> void {
  grad = pwie::Vector::Zero(2);
  grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
  grad[1]  = 200 * (x[1] - x[0] * x[0]);
};
auto DDrosenbrock = [](const pwie::Vector x, pwie::Matrix &hes) -> void {
  hes = pwie::Matrix::Zero(x.rows(), x.rows());
  hes(0, 0) = 1200 * x[0] * x[0] - 400 * x[1] + 1;
  hes(1, 0) = -400 * x[0];
  hes(0, 1) = -400 * x[0];
  hes(1, 1) = 200;
};

#define SOLVE_0stOrder(sol,func,a,b,fx)   pwie::Vector x(2);x(0) = a;x(1) = b; sol solver; solver.solve(x, func);                     EXPECT_NEAR(fx, func(x), PRECISION);
#define SOLVE_1stOrder(sol,func,a,b,fx)   pwie::Vector x(2);x(0) = a;x(1) = b; sol solver; solver.solve(x, func, D##func);            EXPECT_NEAR(fx, func(x), PRECISION);
#define SOLVE_2ndOrder(sol,func,a,b,fx)   pwie::Vector x(2);x(0) = a;x(1) = b; sol solver; solver.solve(x, func, D##func,  DD##func); EXPECT_NEAR(fx, func(x), PRECISION);

TEST(GradientDescentTest, RosenbrockFar)        { SOLVE_1stOrder(pwie::GradientDescentSolver, rosenbrock, 15.0, 8.0, 0.0) }
TEST(GradientDescentTest, RosenbrockNear)       { SOLVE_1stOrder(pwie::GradientDescentSolver, rosenbrock, -1.2, 1.0, 0.0) }

TEST(ConjugateGradientTest, RosenbrockFar)      { SOLVE_1stOrder(pwie::ConjugateGradientSolver, rosenbrock, 15.0, 8.0, 0.0) }
TEST(ConjugateGradientTest, RosenbrockNear)     { SOLVE_1stOrder(pwie::ConjugateGradientSolver, rosenbrock, -1.2, 1.0, 0.0) }
TEST(ConjugateGradientTest, RosenbrockFarNear)  { SOLVE_1stOrder(pwie::ConjugateGradientSolver, rosenbrock, -1.2, 100.0, 0.0) }

TEST(NewtonDescentTest, RosenbrockFar)          { SOLVE_2ndOrder(pwie::NewtonDescentSolver, rosenbrock, 15.0, 8.0, 0.0) }
TEST(NewtonDescentTest, RosenbrockNear)         { SOLVE_2ndOrder(pwie::NewtonDescentSolver, rosenbrock, -1.2, 1.0, 0.0) }
TEST(NewtonDescentTest, RosenbrockFarNear)      { SOLVE_2ndOrder(pwie::NewtonDescentSolver, rosenbrock, -1.2, 100.0, 0.0) }

TEST(BfgsTest, RosenbrockFar)                   { SOLVE_1stOrder(pwie::BfgsSolver, rosenbrock, 15.0, 8.0, 0.0) }
TEST(BfgsTest, RosenbrockNear)                  { SOLVE_1stOrder(pwie::BfgsSolver, rosenbrock, -1.2, 1.0, 0.0) }
TEST(BfgsTest, RosenbrockFarNear)               { SOLVE_1stOrder(pwie::BfgsSolver, rosenbrock, -1.2, 100.0, 0.0) }

TEST(LbfgsTest, RosenbrockFar)                  { SOLVE_1stOrder(pwie::LbfgsSolver, rosenbrock, 15.0, 8.0, 0.0) }
TEST(LbfgsTest, RosenbrockNear)                 { SOLVE_1stOrder(pwie::LbfgsSolver, rosenbrock, -1.2, 1.0, 0.0) }
TEST(LbfgsTest, RosenbrockFarNear)              { SOLVE_1stOrder(pwie::LbfgsSolver, rosenbrock, -1.2, 100.0, 0.0) }

TEST(LbfgsbTest, DISABLED_RosenbrockFar)        { SOLVE_1stOrder(pwie::LbfgsbSolver, rosenbrock, 15.0, 8.0, 0.0) }
TEST(LbfgsbTest, DISABLED_RosenbrockNear)       { SOLVE_1stOrder(pwie::LbfgsbSolver, rosenbrock, -1.2, 1.0, 0.0) }

int main (int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}