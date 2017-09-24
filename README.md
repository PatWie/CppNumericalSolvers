CppOptimizationLibrary
=================================================================

[![Build Status](https://api.travis-ci.org/PatWie/CppNumericalSolvers.svg?branch=master)](http://travis-ci.org/PatWie/CppNumericalSolvers)

Get this Library
-----------

````bash
    git clone https://github.com/PatWie/CppNumericalSolvers.git
    cd CppNumericalSolvers/
    bazel run verify
````

About
-----------
Have you ever looked for a c++ version of MATLAB's *fminsearch*, which is easy to use without adding tons of dependencies and without editing many setting-structs? This project exactly addresses this issue by providing a *header-only* library without dependencies. All solvers are written from scratch, which means they do not represent the current state-of-the-art implementation with all tricky optimizations (at least for now). But they are very easy to use. Want a full example?

````cpp
    class Rosenbrock : public Problem<double> {
      public:
        double value(const Vector<double> &x) {
            const double t1 = (1 - x[0]);
            const double t2 = (x[1] - x[0] * x[0]);
            return   t1 * t1 + 100 * t2 * t2;
        }
    };
    int main(int argc, char const *argv[]) {
        Rosenbrock f;
        Vector<double> x(2); x << -1, 2;
        BfgsSolver<double> solver;
        solver.minimize(f, x);
        std::cout << "argmin      " << x.transpose() << std::endl;
        std::cout << "f in argmin " << f(x) << std::endl;
        return 0;
    }
````
To use another solver, simply replace `BfgsSolver` by another name.
Supported solvers are:

- gradient descent solver (GradientDescentSolver)
- conjugate gradient descent solver (ConjugatedGradientDescentSolver)
- Newton descent solver (NewtonDescentSolver)
- BFGS solver (BfgsSolver)
- L-BFGS solver (LbfgsSolver)
- L-BFGS-B solver (LbfgsbSolver)
- CMAes solver (CMAesSolver)
- Nelder-Mead solver (NelderMeadSolver)

These solvers are tested on the Rosenbrock function from multiple difficult starting points by unit tests using the Google Testing Framework. And yes, you can use them directly in MATLAB.
Additional benchmark functions are *Beale, GoldsteinPrice, Booth, Matyas, Levi*. Note, not all solvers are equivalently good at all problems.

For checking your gradient this library uses high-order central difference. Study the examples for more information about including box-constraints and gradient-information.

Install
-----------

Note, this library is header-only, so you just need to add `include/cppoptlib` to your project without compiling anything and without adding further dependencies. We ship some examples for demonstration purposes and use [bazel](https://bazel.build/) to compile these examples and unittests. The latest commit using CMake is da314c6581d076e0dbadacdd263aefe4d06a2397.

When using the MATLAB-binding, you need to compile the mex-file. Hereby, open Matlab and run `make.m` inside the MATLAB folder once.

Extensive Introduction
-----------

There are currently two ways to use this library: directly in your C++ code or in MATLAB by calling the provided mex-File.

## C++

There are several examples within the `src/examples` directory. These are built into `build/bin/examples` during `make all`.
Checkout `rosenbrock.cpp`. Your objective and gradient computations should be stored into a tiny class. The most simple usage is
````cpp
    class YourProblem : public Problem<double> {
      double value(const Vector<double> &x) {}
    }
````
In contrast to previous versions of this library, I switched to classes instead of lambda function. If you poke the examples, you will notice that this is much easier to write and understand. The only method a problem has to provide is the `value` member, which returns the value of the objective function.
For most solvers it should be useful to implement the gradient computation, too. Otherwise the library internally will use finite difference for gradient computations (which is definitely unstable and slow!).
````cpp
    class YourProblem : public Problem<double> {
      double value(const Vector<double> &x) {}
      void gradient(const Vector<double> &x, Vector<double> &grad) {}
    }
````
Notice, the gradient is passed by reference!
After defining the problem it can be initialized in your code by:
````cpp
    // init problem
    YourProblem f;
    // test your gradient ONCE
    bool probably_correct = f.checkGradient(x);
````
By convention, a solver minimizes a given objective function starting in `x`
````cpp
    // choose a solver
    BfgsSolver<double> solver;
    // and minimize the function
    solver.minimize(f, x);
    double minValue = f(x);
````
For convenience there are some typedefs:
````cpp
    cppoptlib::Vector<T> is a column vector Eigen::Matrix<T, Eigen::Dynamic, 1>;
    cppoptlib::Matrix<T> is a matrix        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
````
### full example
````cpp
    #include <iostream>
    #include "../../include/cppoptlib/meta.h"
    #include "../../include/cppoptlib/problem.h"
    #include "../../include/cppoptlib/solver/bfgssolver.h"

    using namespace cppoptlib;
    using Eigen::VectorXd;

    class Rosenbrock : public Problem<double> {
      public:
        using typename cppoptlib::Problem<double>::Scalar;
        using typename cppoptlib::Problem<double>::TVector;

        double value(const TVector &x) {
            const double t1 = (1 - x[0]);
            const double t2 = (x[1] - x[0] * x[0]);
            return   t1 * t1 + 100 * t2 * t2;
        }
        void gradient(const TVector &x, TVector &grad) {
            grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
            grad[1]  = 200 * (x[1] - x[0] * x[0]);
        }
    };

    int main(int argc, char const *argv[]) {
        Rosenbrock f;
        BfgsSolver<Rosenbrock> solver;
        VectorXd x(2); x << -1, 2;
        solver.minimize(f, x);
        std::cout << "argmin      " << x.transpose() << std::endl;
        std::cout << "f in argmin " << f(x) << std::endl;
        return 0;
    }
````
### using box-constraints

The `L-BFGS-B` algorithm allows to optimize functions with box-constraints, i.e., `min_x f(x) s.t. a <= x <= b` for some `a, b`. Given a `BoundedProblem`-class you can enter these constraints by
````cpp
    cppoptlib::LbfgsbSolver<T> f;
    f.setLowerBound(Vector<double>::Zero(DIM));
    f.setUpperBound(Vector<double>::Ones(DIM)*5);
````
If you do not specify a bound, the algorithm will assume the unbounded case, eg.
````cpp
    cppoptlib::YourProblem<T> f;
    f.setLowerBound(Vector<double>::Zero(DIM));
````
will optimize in x s.t. `0 <= x`.

## within MATLAB

Simply create a function file for the objective and replace `fminsearch` or `fminunc` with `cppoptlib`. If you want to use symbolic gradient or hessian information see file `example.m` for details. A basic example would be:
````matlab
    x0 = [-1,2]';
    [fx,x] = cppoptlib(x0, @rosenbrock,'gradient',@rosenbrock_grad,'solver','bfgs');
    fx     = cppoptlib(x0, @rosenbrock);
    fx     = fminsearch(x0, @rosenbrock);
````
Even optimizing without any gradient information this library outperforms optimization routines from MATLAB on some problems.

# Benchmarks

Currently, not all solvers are equally good at all objective functions. The file `src/test/benchmark.cpp` contains some challenging objective functions which are tested by each provided solver. Note, MATLAB will also fail at some objective functions.

# Contribute

Make sure that `python lint.py` does not display any errors and check if travis is happy. It would be great, if some of the Forks-Owner are willing to make pull-request.

[eigen3]: http://eigen.tuxfamily.org/
[bazel]: https://bazel.build/
[matlab]: http://www.mathworks.de/products/matlab/
