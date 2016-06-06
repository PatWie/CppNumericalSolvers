#undef NDEBUG

#include <iostream>
#include <functional>
#include <list>
#include "../../gtest/googletest/include/gtest/gtest.h"
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/boundedproblem.h"
#include "../../include/cppoptlib/solver/gradientdescentsolver.h"
#include "../../include/cppoptlib/solver/conjugatedgradientdescentsolver.h"
#include "../../include/cppoptlib/solver/newtondescentsolver.h"
#include "../../include/cppoptlib/solver/bfgssolver.h"
#include "../../include/cppoptlib/solver/lbfgssolver.h"
#include "../../include/cppoptlib/solver/lbfgsbsolver.h"
#include "../../include/cppoptlib/solver/cmaessolver.h"
#include "../../include/cppoptlib/solver/neldermeadsolver.h"
#define PRECISION 1e-4
using namespace cppoptlib;


typedef ::testing::Types <float, double> MyTypeList;


// situation where only have to objective function
template<typename Scalar>
class RosenbrockValue : public BoundedProblem<Scalar, 2> {
  public:
    using typename Problem<Scalar, 2>::TVector;
    
    Scalar value(const TVector &x) {
        const Scalar t1 = (1 - x[0]);
        const Scalar t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    }

};

// now we add the information about the gradient
template<typename Scalar>
class RosenbrockGradient : public RosenbrockValue<Scalar> {
  public:
    using typename RosenbrockValue<Scalar>::TVector;
    
    /*T value(const Vector<T> &x) {
        const T t1 = (1 - x[0]);
        const T t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    }*/

    void gradient(const TVector &x, TVector &grad) {
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1]  = 200 * (x[1] - x[0] * x[0]);
    }

};

// now we add the information about the hessian
template<typename Scalar>
class RosenbrockFull : public RosenbrockGradient<Scalar> {
  public:
  using typename RosenbrockGradient<Scalar>::TVector;
  using typename Problem<Scalar, 2>::THessian;
    /*T value(const Vector<T> &x) {
        const T t1 = (1 - x[0]);
        const T t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    }

    void gradient(const Vector<T> &x, Vector<T> &grad) {
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1]  = 200 * (x[1] - x[0] * x[0]);
    }*/

    void hessian(const TVector &x, THessian &hessian) {
        hessian(0, 0) = 1200 * x[0] * x[0] - 400 * x[1] + 1;
        hessian(0, 1) = -400 * x[0];
        hessian(1, 0) = -400 * x[0];
        hessian(1, 1) = 200;
    }
};

template <class T> class GradientDescentTest : public testing::Test{};
template <class T> class ConjugatedGradientDescentTest : public testing::Test{};
template <class T> class NewtonDescentTest : public testing::Test{};
template <class T> class BfgsTest : public testing::Test{};
template <class T> class LbfgsTest : public testing::Test{};
template <class T> class LbfgsbTest : public testing::Test{};
template <class T> class CMAesTest : public testing::Test{};
template <class T> class NelderMeadTest : public testing::Test{};
template <class T> class CentralDifference : public testing::Test{};

#define SOLVE_PROBLEM( sol, func, a, b, fx ) \
    typedef func<TypeParam> TProblem;\
    TProblem f;\
    typename TProblem::TVector x; x << a, b;\
    sol<TProblem> solver;\
    solver.minimize(f, x);\
    EXPECT_NEAR(fx, f(x), PRECISION);
#define SOLVE_PROBLEM_F( sol, func, a, b, fx ) \
    typedef func<float> TProblem;\
    TProblem f;\
    typename TProblem::TVector x; x << a, b;\
    sol<TProblem> solver;\
    solver.minimize(f, x);\
    EXPECT_NEAR(fx, f(x), PRECISION);
#define SOLVE_PROBLEM_D( sol, func, a, b, fx ) \
    typedef func<double> TProblem;\
    TProblem f;\
    typename TProblem::TVector x; x << a, b;\
    sol<TProblem> solver;\
    solver.minimize(f, x);\
    EXPECT_NEAR(fx, f(x), PRECISION);

TYPED_TEST_CASE(GradientDescentTest, MyTypeList);
TYPED_TEST_CASE(ConjugatedGradientDescentTest, MyTypeList);
TYPED_TEST_CASE(NewtonDescentTest, MyTypeList);
TYPED_TEST_CASE(BfgsTest, MyTypeList);
TYPED_TEST_CASE(LbfgsTest, MyTypeList);
TYPED_TEST_CASE(LbfgsbTest, MyTypeList);
TYPED_TEST_CASE(CMAesTest, MyTypeList);
TYPED_TEST_CASE(NelderMeadTest, MyTypeList);
TYPED_TEST_CASE(CentralDifference, MyTypeList);

// only gradient information
TEST(GradientDescentTest, RosenbrockFarValue)                      { SOLVE_PROBLEM_D(cppoptlib::GradientDescentSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(GradientDescentTest, RosenbrockNearValue)                     { SOLVE_PROBLEM_D(cppoptlib::GradientDescentSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TEST(ConjugatedGradientDescentTest, RosenbrockFarValue)            { SOLVE_PROBLEM_D(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(ConjugatedGradientDescentTest, RosenbrockNearValue)           { SOLVE_PROBLEM_D(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TEST(ConjugatedGradientDescentTest, RosenbrockMixValue)            { SOLVE_PROBLEM_D(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockValue, -1.2, 100.0, 0.0) }
TEST(BfgsTest, RosenbrockFarValue)                                 { SOLVE_PROBLEM_D(cppoptlib::BfgsSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(BfgsTest, RosenbrockNearValue)                                { SOLVE_PROBLEM_D(cppoptlib::BfgsSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TEST(BfgsTest, RosenbrockMixValue)                                 { SOLVE_PROBLEM_D(cppoptlib::BfgsSolver,RosenbrockValue, -1.2, 100.0, 0.0) }
TEST(LbfgsTest, RosenbrockFarValue)                                { SOLVE_PROBLEM_D(cppoptlib::LbfgsSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(LbfgsTest, RosenbrockNearValue)                               { SOLVE_PROBLEM_D(cppoptlib::LbfgsSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TEST(LbfgsTest, RosenbrockMixValue)                                { SOLVE_PROBLEM_D(cppoptlib::LbfgsSolver,RosenbrockValue, -1.2, 100.0, 0.0) }
TEST(LbfgsbTest, RosenbrockFarValue)                               { SOLVE_PROBLEM_D(cppoptlib::LbfgsbSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(LbfgsbTest, RosenbrockNearValue)                              { SOLVE_PROBLEM_D(cppoptlib::LbfgsbSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TEST(LbfgsbTest, RosenbrockMixValue)                               { SOLVE_PROBLEM_D(cppoptlib::LbfgsbSolver,RosenbrockValue, -1.2, 100.0, 0.0) }
TEST(CMAesTest, RosenbrockFarValue)                                { SOLVE_PROBLEM_D(cppoptlib::CMAesSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TEST(CMAesTest, RosenbrockNearValue)                               { SOLVE_PROBLEM_D(cppoptlib::CMAesSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
//TEST(CMAesTest, RosenbrockMixValue)                                { SOLVE_PROBLEM_D(cppoptlib::CMAesSolver,RosenbrockValue, -1.2, 100.0, 0.0) }
TYPED_TEST(NelderMeadTest, RosenbrockFarValue)                     { SOLVE_PROBLEM(cppoptlib::NelderMeadSolver,RosenbrockValue, 15.0, 8.0, 0.0) }
TYPED_TEST(NelderMeadTest, RosenbrockNearValue)                    { SOLVE_PROBLEM(cppoptlib::NelderMeadSolver,RosenbrockValue, -1.0, 2.0, 0.0) }
TYPED_TEST(NelderMeadTest, RosenbrockMixValue)                     { SOLVE_PROBLEM(cppoptlib::NelderMeadSolver,RosenbrockValue, -1.2, 100.0, 0.0) }

// gradient information ( Hessian for newton descent)
TYPED_TEST(GradientDescentTest, RosenbrockFarGradient)             { SOLVE_PROBLEM(cppoptlib::GradientDescentSolver,RosenbrockGradient, 15.0, 8.0, 0.0) }
TYPED_TEST(GradientDescentTest, RosenbrockNearGradient)            { SOLVE_PROBLEM(cppoptlib::GradientDescentSolver,RosenbrockGradient, -1.0, 2.0, 0.0) }
TYPED_TEST(ConjugatedGradientDescentTest, RosenbrockFarGradient)   { SOLVE_PROBLEM(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockGradient, 15.0, 8.0, 0.0) }
TYPED_TEST(ConjugatedGradientDescentTest, RosenbrockNearGradient)  { SOLVE_PROBLEM(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockGradient, -1.0, 2.0, 0.0) }
TYPED_TEST(ConjugatedGradientDescentTest, RosenbrockMixGradient)   { SOLVE_PROBLEM(cppoptlib::ConjugatedGradientDescentSolver,RosenbrockGradient, -1.2, 100.0, 0.0) }
TYPED_TEST(NewtonDescentTest, RosenbrockFarFull)                   { SOLVE_PROBLEM(cppoptlib::NewtonDescentSolver,RosenbrockFull, 15.0, 8.0, 0.0) }
TYPED_TEST(NewtonDescentTest, RosenbrockNearFull)                  { SOLVE_PROBLEM(cppoptlib::NewtonDescentSolver,RosenbrockFull, -1.0, 2.0, 0.0) }
TYPED_TEST(NewtonDescentTest, RosenbrockMixFull)                   { SOLVE_PROBLEM(cppoptlib::NewtonDescentSolver,RosenbrockFull, -1.2, 100.0, 0.0) }
TYPED_TEST(BfgsTest, RosenbrockFarFull)                            { SOLVE_PROBLEM(cppoptlib::BfgsSolver,RosenbrockFull, 15.0, 8.0, 0.0) }
TYPED_TEST(BfgsTest, RosenbrockNearFull)                           { SOLVE_PROBLEM(cppoptlib::BfgsSolver,RosenbrockFull, -1.0, 2.0, 0.0) }
TYPED_TEST(BfgsTest, RosenbrockMixFull)                            { SOLVE_PROBLEM(cppoptlib::BfgsSolver,RosenbrockFull, -1.2, 100.0, 0.0) }
TYPED_TEST(LbfgsTest, RosenbrockFarFull)                           { SOLVE_PROBLEM(cppoptlib::LbfgsSolver,RosenbrockFull, 15.0, 8.0, 0.0) }
TYPED_TEST(LbfgsTest, RosenbrockNearFull)                          { SOLVE_PROBLEM(cppoptlib::LbfgsSolver,RosenbrockFull, -1.0, 2.0, 0.0) }
TYPED_TEST(LbfgsTest, RosenbrockMixFull)                           { SOLVE_PROBLEM(cppoptlib::LbfgsSolver,RosenbrockFull, -1.2, 100.0, 0.0) }
TYPED_TEST(LbfgsbTest, RosenbrockFarFull)                          { SOLVE_PROBLEM(cppoptlib::LbfgsbSolver,RosenbrockFull, 15.0, 8.0, 0.0) }
TYPED_TEST(LbfgsbTest, RosenbrockNearFull)                         { SOLVE_PROBLEM(cppoptlib::LbfgsbSolver,RosenbrockFull, -1.0, 2.0, 0.0) }

TEST(LbfgsbTest, RosenbrockMixFull)                          { SOLVE_PROBLEM_D(cppoptlib::LbfgsbSolver,RosenbrockFull, -1.2, 100.0, 0.0) }

TYPED_TEST(CMAesTest, RosenbrockFarFull)                           { SOLVE_PROBLEM(cppoptlib::CMAesSolver,RosenbrockFull, 15.0, 8.0, 0.0) }
TYPED_TEST(CMAesTest, RosenbrockNearFull)                          { SOLVE_PROBLEM(cppoptlib::CMAesSolver,RosenbrockFull, -1.0, 2.0, 0.0) }
//TYPED_TEST(CMAesTest, RosenbrockMixFull)                           { SOLVE_PROBLEM(cppoptlib::CMAesSolver,RosenbrockFull, -1.2, 100.0, 0.0) }

TYPED_TEST(CentralDifference, Gradient){
    // simple function y <- 3*a-b
    class Func : public Problem<TypeParam, 2> {
      public:
        using typename Problem<TypeParam, 2>::TVector;
        TypeParam value(const TVector &x) {
            return 3*x[0]-x[1];
        }
    };
    typename Func::TVector x0;
    x0(0) = 0;
    x0(1) = 0;

    Func f;
    typename Func::TVector grad;
    // check from fast/bad to slower/better approximation of the gradient
    for (int accuracy = 0; accuracy < 4; ++accuracy)
    {
        f.finiteGradient(x0, grad, accuracy);
        EXPECT_NEAR(grad(0), 3, PRECISION);
        EXPECT_NEAR(grad(1), -1, PRECISION);
    }
}

TYPED_TEST(CentralDifference, Hessian){
    // simple function y <- 3*a^2-a*b
    class Func : public Problem<TypeParam, 2> {
      public:
        using typename Problem<TypeParam, 2>::TVector;
        TypeParam value(const TVector &x) {
            return 3*x[0]*x[0]-x[1]*x[0];
        }
    };
    typename Func::TVector x0;
    x0(0) = 0;
    x0(1) = 0;

    Func f;
    typename Func::THessian hessian;

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