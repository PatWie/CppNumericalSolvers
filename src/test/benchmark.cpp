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
#include "../../include/cppoptlib/solver/lbfgsbsolver.h"
#include "../../include/cppoptlib/solver/cmaessolver.h"
#define PRECISION 1e-4
#define PI 3.14159265358979323846
using namespace cppoptlib;


// collection of benchmark functions
// ----------------------------------------------------------------------------------
template<typename T>
class Rosenbrock : public Problem<T> {
 public:

  T value(const Vector<T> &x) {
    const size_t n = x.rows();
    T sum = 0;

    for (int i = 0; i < n - 1; ++i) {
      const T t1 = (1 - x[i]);
      const T t2 = (x[i + 1] - x[i] * x[i]);
      sum += t1 * t1 + 100 * t2 * t2;
    }
    return sum;
  }

};

template<typename T>
class Beale : public Problem<T> {
 public:
  T value(const Vector<T> &xx) {
    const T x = xx[0];
    const T y = xx[1];
    const T t1 = (1.5 - x + x * y);
    const T t2 = (2.25 - x + x * y * y);
    const T t3 = (2.623 - x + x * y * y * y);
    return  t1 * t1 + t2 * t2 + t3 * t3;
  }

  void gradient(const Vector<T> &xx, Vector<T> &grad) {
    const T x = xx[0];
    const T y = xx[1];

    grad[0] = 2. * (0.15e1 - x + x * y) * (-1. + y) + 2. * (0.225e1 - x + x * y * y) * (y * y - 1.) + 2. * (0.2623e1 - x + x * y*y*y) * (y*y*y - 1.);
    grad[1] = 2. * (0.15e1 - x + x * y) * x + 0.4e1 * (0.225e1 - x + x * y * y) * x * y + 0.6e1 * (0.2623e1 - x + x * y*y*y) * x * y * y;
  }
};

template<typename T>
class GoldsteinPrice  : public Problem<T> {
 public:
  T value(const Vector<T> &xx) {
    const T x = xx[0];
    const T y = xx[1];

    const T t1 = x+y+1;
    const T t2 = 19-14*x+3*x*x-14*y+6*x*y+3*y*y;
    const T t3 = 2*x-3*y;
    const T t4 = 18-32*x+12*x*x+48*y-36*x*y+27*y*y;

    return  (1+t1*t1*t2)*(30+t3*t3*t4);
  }
};

template<typename T>
class Booth  : public Problem<T> {
 public:
  T value(const Vector<T> &xx) {
    const T x = xx[0];
    const T y = xx[1];

    const T t1 = x+2*y-7;
    const T t2 = 2*x+y-5;;

    return  t1*t1+t2*t2;
  }

  void gradient(const Vector<T> &xx, Vector<T> &grad) {
    const T x = xx[0];
    const T y = xx[1];

    grad[0] = 2*(x+2*y-7) + 2*(2*x+y-5)*2;
    grad[1] = 2*(x+2*y-7)*2 + 2*(2*x+y-5);

  }
};

template<typename T>
class Matyas   : public Problem<T> {
 public:
  T value(const Vector<T> &xx) {
    const T x = xx[0];
    const T y = xx[1];
    return  0.26*(x*x+y*y)-0.48*x*y;
  }

  void gradient(const Vector<T> &xx, Vector<T> &grad) {
    const T x = xx[0];
    const T y = xx[1];

    grad[0] = 0.26*2*x-0.48*y;
    grad[1] = 0.26*2*y-0.48*x;

  }

};

template<typename T>
class Levi   : public Problem<T> {
 public:
  T value(const Vector<T> &xx) {
    const T x = xx[0];
    const T y = xx[1];
    
    return sin(3*PI*x)*sin(3*PI*x)+(x-1)*(x-1)*(1+sin(3*PI*y)*sin(3*PI*y)) +(y-1)*(y-1)*(1+sin(2*PI*y)*sin(2*PI*y)); 
  }

  void gradient(const Vector<T> &xx, Vector<T> &grad) {
    const T x = xx[0];
    const T y = xx[1];

    grad[0] = 0.6e1 * sin(3. * PI * x) * cos(3. * PI * x) * PI + 2. * (x - 1.) * (1. + pow(sin(3. * PI * y), 2.));
    grad[1] = 0.6e1 * (double) (int) pow((double) (x - 1), (double) 2) * sin(3. * PI * y) * cos(3. * PI * y) * PI + 2. * (y - 1.) * (1. + pow(sin(2. * PI * y), 2.)) + 0.4e1 * pow(y - 1., 2.) * sin(2. * PI * y) * cos(2. * PI * y) * PI;


  }


};




// define test body
// ----------------------------------------------------------------------------------
#define CHECKDIFF(func, a,b)  TEST(GradientTest, func) {                                                                 \
                                Vector<double> x(2);                                                                     \
                                x(0) = a;                                                                                \
                                x(1) = b;                                                                                \
                                func<double> f;                                                                          \
                                EXPECT_EQ(true,f.checkGradient(x));                                                      \
                              }
                                              
                                              

#define BENCH2(sol, func, a,b, fx, y0, y1 ,PREC)   TEST(D2Functions##sol, func) {            \
                                                Vector<double> x(2);               \
                                                x(0) = a;                          \
                                                x(1) = b;                          \
                                                func<double> f;                    \
                                                sol<double> solver;                \
                                                solver.minimize(f, x);             \
                                                EXPECT_NEAR(fx, f(x), PREC);  \
                                                EXPECT_NEAR(y0, x(0), PREC);  \
                                                EXPECT_NEAR(y1, x(1), PREC);  \
                                              }                                  

// optimize and test all different function
// ----------------------------------------------------------------------------------

#define BENCHSOVLER(sol) BENCH2(sol, Rosenbrock,          -1, 2,   0, 1, 1, PRECISION);   \
                         BENCH2(sol, Beale,               -1, 2,   0, 3, 0.5, 0.1); \
                         BENCH2(sol, GoldsteinPrice,      -1, 1.5, 3, 0, -1, PRECISION);  \
                         BENCH2(sol, Booth,               -4, 3.7, 0, 1, 3, PRECISION);   \
                         BENCH2(sol, Matyas,              -4, 3.7, 0, 0, 0, PRECISION);   \
                         BENCH2(sol, Levi,                -4, 3.7, 0, 1, 1, PRECISION);  



CHECKDIFF(Rosenbrock,          -1, 2   )
CHECKDIFF(Beale,               -1, 2   )
CHECKDIFF(GoldsteinPrice,      -1, 1.5 )
CHECKDIFF(Booth,               -4, 3.7 )
CHECKDIFF(Matyas,              -4, 3.7 )
CHECKDIFF(Levi,                -4, 3.7 )

BENCHSOVLER(GradientDescentSolver)
BENCHSOVLER(ConjugatedGradientDescentSolver)
BENCHSOVLER(BfgsSolver)
BENCHSOVLER(LbfgsSolver)
BENCHSOVLER(LbfgsbSolver)
BENCHSOVLER(CMAesSolver)


int main (int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}