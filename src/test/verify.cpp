#include <iostream>
#include <functional>
#include <list>
#include "../../include/gtest/gtest.h"
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/gradientdescentsolver.h"
#include "../../include/cppoptlib/solver/conjugatedgradientdescentsolver.h"
#include "../../include/cppoptlib/solver/newtondescentsolver.h"
#include "../../include/cppoptlib/solver/bfgssolver.h"
#include "../../include/cppoptlib/solver/lbfgssolver.h"
#define PRECISION 1e-4
using namespace cppoptlib;

// situation where only have to objective function
template<typename T>
class RosenbrockValue : public Problem<T> {
  public:
    
    T value(const Vector<T> &x) {
        const T t1 = (1 - x[0]);
        const T t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    }

};

// now we add the information about the gradient
template<typename T>
class RosenbrockGradient : public Problem<T> {
  public:
    
    T value(const Vector<T> &x) {
        const T t1 = (1 - x[0]);
        const T t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    }

    void gradient(const Vector<T> &x, Vector<T> &grad) {
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1]  = 200 * (x[1] - x[0] * x[0]);
    }

};

// now we add the information about the hessian
template<typename T>
class RosenbrockFull : public Problem<T> {
  public:
    
    T value(const Vector<T> &x) {
        const T t1 = (1 - x[0]);
        const T t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    }

    void gradient(const Vector<T> &x, Vector<T> &grad) {
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1]  = 200 * (x[1] - x[0] * x[0]);
    }

    void hessian(const Vector<T> &x, Matrix<T> & hessian) {
        hessian(0, 0) = 1200 * x[0] * x[0] - 400 * x[1] + 1;
        hessian(0, 1) = -400 * x[0];
        hessian(1, 0) = -400 * x[0];
        hessian(1, 1) = 200;
    }
};



#define SOLVE_PROBLEM(sol, func, a,b, fx )    Vector<double> x(2);x(0) = a;x(1) = b; func<double> f;    sol<double> solver; solver.minimize(f, x); EXPECT_NEAR(fx, f(x), PRECISION);

TEST(GradientDescentTest, RosenbrockFarValue)                { SOLVE_PROBLEM(cppoptlib::GradientDescentSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(GradientDescentTest, RosenbrockNearValue)               { SOLVE_PROBLEM(cppoptlib::GradientDescentSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TEST(GradientDescentTest, RosenbrockFarGradient)             { SOLVE_PROBLEM(cppoptlib::GradientDescentSolver,RosenbrockGradient, 15.0, 8.0, 0.0) }
TEST(GradientDescentTest, RosenbrockNearGradient)            { SOLVE_PROBLEM(cppoptlib::GradientDescentSolver,RosenbrockGradient, -1.0, 2.0, 0.0) }

TEST(ConjugatedGradientDescentTest, RosenbrockFarValue)      { SOLVE_PROBLEM(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(ConjugatedGradientDescentTest, RosenbrockNearValue)     { SOLVE_PROBLEM(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TEST(ConjugatedGradientDescentTest, RosenbrockMixValue)      { SOLVE_PROBLEM(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockValue, -1.2, 100.0, 0.0) }
TEST(ConjugatedGradientDescentTest, RosenbrockFarGradient)   { SOLVE_PROBLEM(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockGradient, 15.0, 8.0, 0.0) }
TEST(ConjugatedGradientDescentTest, RosenbrockNearGradient)  { SOLVE_PROBLEM(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockGradient, -1.0, 2.0, 0.0) }
TEST(ConjugatedGradientDescentTest, RosenbrockMixGradient)   { SOLVE_PROBLEM(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockGradient, -1.2, 100.0, 0.0) }

TEST(NewtonDescentTest, RosenbrockFarFull)                   { SOLVE_PROBLEM(cppoptlib::NewtonDescentSolver,RosenbrockFull, 15.0, 8.0, 0.0) }
TEST(NewtonDescentTest, RosenbrockNearFull)                  { SOLVE_PROBLEM(cppoptlib::NewtonDescentSolver,RosenbrockFull, -1.0, 2.0, 0.0) }
TEST(NewtonDescentTest, RosenbrockMixFull)                   { SOLVE_PROBLEM(cppoptlib::NewtonDescentSolver,RosenbrockFull, -1.2, 100.0, 0.0) }

TEST(BfgsTest, RosenbrockFarValue)                           { SOLVE_PROBLEM(cppoptlib::BfgsSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(BfgsTest, RosenbrockNearValue)                          { SOLVE_PROBLEM(cppoptlib::BfgsSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TEST(BfgsTest, RosenbrockMixValue)                           { SOLVE_PROBLEM(cppoptlib::BfgsSolver,RosenbrockValue, -1.2, 100.0, 0.0) }
TEST(BfgsTest, RosenbrockFarFull)                            { SOLVE_PROBLEM(cppoptlib::BfgsSolver,RosenbrockFull, 15.0, 8.0, 0.0) }
TEST(BfgsTest, RosenbrockNearFull)                           { SOLVE_PROBLEM(cppoptlib::BfgsSolver,RosenbrockFull, -1.0, 2.0, 0.0) }
TEST(BfgsTest, RosenbrockMixFull)                            { SOLVE_PROBLEM(cppoptlib::BfgsSolver,RosenbrockFull, -1.2, 100.0, 0.0) }

TEST(LbfgsTest, RosenbrockFarValue)                          { SOLVE_PROBLEM(cppoptlib::LbfgsSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(LbfgsTest, RosenbrockNearValue)                         { SOLVE_PROBLEM(cppoptlib::LbfgsSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TEST(LbfgsTest, RosenbrockMixValue)                          { SOLVE_PROBLEM(cppoptlib::LbfgsSolver,RosenbrockValue, -1.2, 100.0, 0.0) }
TEST(LbfgsTest, RosenbrockFarFull)                           { SOLVE_PROBLEM(cppoptlib::LbfgsSolver,RosenbrockFull, 15.0, 8.0, 0.0) }
TEST(LbfgsTest, RosenbrockNearFull)                          { SOLVE_PROBLEM(cppoptlib::LbfgsSolver,RosenbrockFull, -1.0, 2.0, 0.0) }
TEST(LbfgsTest, RosenbrockMixFull)                           { SOLVE_PROBLEM(cppoptlib::LbfgsSolver,RosenbrockFull, -1.2, 100.0, 0.0) }

// TEST(LbfgsbTest, DISABLED_RosenbrockFarValue)                { SOLVE_PROBLEM(cppoptlib::LbfgsbSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
// TEST(LbfgsbTest, DISABLED_RosenbrockNearValue)               { SOLVE_PROBLEM(cppoptlib::LbfgsbSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
// TEST(LbfgsbTest, DISABLED_RosenbrockMixValue)                { SOLVE_PROBLEM(cppoptlib::LbfgsbSolver,RosenbrockValue, -1.2, 100.0, 0.0) }
// TEST(LbfgsbTest, DISABLED_RosenbrockFarFull)                 { SOLVE_PROBLEM(cppoptlib::LbfgsbSolver,RosenbrockFull, 15.0, 8.0, 0.0) }
// TEST(LbfgsbTest, DISABLED_RosenbrockNearFull)                { SOLVE_PROBLEM(cppoptlib::LbfgsbSolver,RosenbrockFull, -1.0, 2.0, 0.0) }
// TEST(LbfgsbTest, DISABLED_RosenbrockMixFull)                 { SOLVE_PROBLEM(cppoptlib::LbfgsbSolver,RosenbrockFull, -1.2, 100.0, 0.0) }

TEST(CentralDifference, Gradient){
    // simple function y <- 3*a-b
    class Func : public Problem<double> {
      public:
        double value(const Vector<double> &x) {
            return 3*x[0]-x[1];
        }
    };
    Vector<double> x0(2);
    x0(0) = 0;
    x0(1) = 0;

    Func f;
    Vector<double> grad(2);
    // check from fast/bad to slower/better approximation of the gradient
    for (int accuracy = 0; accuracy < 4; ++accuracy)
    {
        f.finiteGradient(x0, grad, accuracy);
        EXPECT_NEAR(grad(0), 3, PRECISION);
        EXPECT_NEAR(grad(1), -1, PRECISION);
    }
}

TEST(CentralDifference, Hessian){
    // simple function y <- 3*a^2-a*b
    class Func : public Problem<double> {
      public:
        double value(const Vector<double> &x) {
            return 3*x[0]*x[0]-x[1]*x[0];
        }
    };
    Vector<double> x0(2);
    x0(0) = 0;
    x0(1) = 0;

    Func f;
    Matrix<double> hessian(2,2);

    // check using fast version
    f.finiteHessian(x0, hessian);
    EXPECT_NEAR(hessian(0,0), 6, PRECISION);
    EXPECT_NEAR(hessian(1,0), -1, PRECISION);
    EXPECT_NEAR(hessian(0,1), -1, PRECISION);
    EXPECT_NEAR(hessian(1,1), 0, PRECISION);

    // check using slow version
    f.finiteHessian(x0, hessian,3);
    EXPECT_NEAR(hessian(0,0), 6, PRECISION);
    EXPECT_NEAR(hessian(1,0), -1, PRECISION);
    EXPECT_NEAR(hessian(0,1), -1, PRECISION);
    EXPECT_NEAR(hessian(1,1), 0, PRECISION);
}

int main (int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}