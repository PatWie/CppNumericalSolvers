CppOptimizationLibrary
=================================================================

[![Build Status](https://api.travis-ci.org/PatWie/CppNumericalSolvers.svg?branch=master)](http://travis-ci.org/PatWie/CppNumericalSolvers)

About
-----------
Have you ever googled for a c++ version of *fminsearch*, which is easy to use without adding tons of dependencies and without edit many setting-structs? This project exactly address this issue by providing a *header-only* library without dependencies. All solvers are written from scratch, which means they do not represent the current state-of-the-art implementation with all tricky optimizations (at least for now). But they are very easy to use. Want a full exampe?

    class Rosenbrock : public Problem<double> {
      public:
        double value(const Vector<double> &x) {
            const double t1 = (1 - x[0]);
            const double t2 = (x[1] - x[0] * x[0]);
            return   t1 * t1 + 100 * t2 * t2;
        }
    };
    int main(int argc, char const *argv[]) {
        Rosenbrock<double> f;
        Vector<double> x(2); x << -1, 2;
        BfgsSolver<double> solver;
        solver.minimize(f, x);
        std::cout << "argmin      " << x.transpose() << std::endl;
        std::cout << "f in argmin " << f(x) << std::endl;
        return 0;
    }

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
Additional benchmark functions are *Beale, GoldsteinPrice, Booth, Matyas, Levi*. Note, not all solver are equivialent good at all problems.

For checking your gradient this library use high-order central difference. Study the examples for more information about including box-constraints and gradient-information.

Install
-----------

Before compiling you need to adjust one path to the [Eigen3][eigen3]-Library (header only), which can by downloaded by a single bashscript, if you haven't Eigen already.

    # download Eigen
    ./get_dependencies.sh

To compile the examples and the unit test just do

    # copy configuration file
    cp CppNumericalSolvers.config.example CppNumericalSolvers.config
    # adjust paths and compiler (supports g++, clang++)
    edit CppNumericalSolvers.config
    # create a new directory
    mkdir build && cd build   
    cmake ..
    # build tests and demo  
    make all    
    # run all tests                
    ./bin/verify  
    # run an example
    ./bin/examples/linearregression    

For using the MATLAB-binding open Matlab and run `make.m` inside the MATLAB folder once.

Extensive Introduction
-----------

There are currently two ways to use this library: directly in your C++ code or in MATLAB by calling the provided mex-File.

## C++ 

There are several examples within the `src/examples` directory. These are build into `buil/bin/examples` during `make all`.
Checkout `rosenbrock.cpp`. Your objective and gradient computations should be stored into a tiny class. The most simple usage is

    class YourProblem : public Problem<double> {
      double value(const Vector<double> &x) {}
    }

In contrast to previous versions of this library, I switched to classes instead of lambda function. If you poke the examples, you will notice that this much easier to write and understand. The only method a problem has to provide is the `value` memeber, which returns the value of the objective function.
For most solvers it should be useful to implement the gradient computation, too. Otherwise the library internally will uses finite difference for gradient computations (which is definetely unstable and slow!).

    class YourProblem : public Problem<double> {
      double value(const Vector<double> &x) {}
      void gradient(const Vector<double> &x, Vector<double> &grad) {}
    }

Notice, the gradient is passed by reference!
After defining the problem it can be initialised in your code by:

    // init problem
    YourProblem f;
    // test your gradient ONCE
    bool probably_correct = f.checkGradient(x);

By convention, a solver minimizes a given objective function starting in `x`

    // choose a solver
    BfgsSolver<double> solver;
    // and minimize the function
    solver.minimize(f, x);
    double minValue = f(x);

For convenience there are some typedefs:

    cppoptlib::Vector<T> is a column vector Eigen::Matrix<T, Eigen::Dynamic, 1>;
    cppoptlib::Matrix<T> is a matrix        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

### full example

    #include <iostream>
    #include "../../include/cppoptlib/meta.h"
    #include "../../include/cppoptlib/problem.h"
    #include "../../include/cppoptlib/solver/bfgssolver.h"
    using namespace cppoptlib;
    class Rosenbrock : public Problem<double> {
      public:
        double value(const Vector<double> &x) {
            const double t1 = (1 - x[0]);
            const double t2 = (x[1] - x[0] * x[0]);
            return   t1 * t1 + 100 * t2 * t2;
        }
        void gradient(const Vector<double> &x, Vector<double> &grad) {
            grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
            grad[1]  = 200 * (x[1] - x[0] * x[0]);
        }
    };
    int main(int argc, char const *argv[]) {
        Rosenbrock<double> f;
        Vector<double> x(2); x << -1, 2;
        BfgsSolver<double> solver;
        solver.minimize(f, x);
        std::cout << "argmin      " << x.transpose() << std::endl;
        std::cout << "f in argmin " << f(x) << std::endl;
        return 0;
    }

### using box-constraints

The `L-BFGS-B` algorithm allows to optimize functions with box-constraints, i.e., `min_x f(x) s.t. a <= x <= b` for some `a, b`. Given a `problem`-class you can enter these constraints by

    cppoptlib::YourProblem<T> f;
    f.setLowerBound(Vector<double>::Zero(DIM));
    f.setUpperBound(Vector<double>::Ones(DIM)*5);

If you do not specify a bound, the algorithm will assume the unbounded case, eg.

    cppoptlib::YourProblem<T> f;
    f.setLowerBound(Vector<double>::Zero(DIM));

will optimize in x s.t. `0 <= x`.

## within MATLAB

Simply create a function file for the objective and replace `fminsearch` or `fminunc` with `cppoptlib`. If you want to use symbolic gradient or hessian information see file `example.m` for details. A basic example would be:

    x0 = [-1,2]';
    [fx,x] = cppoptlib(x0, @rosenbrock,'gradient',@rosenbrock_grad,'solver','bfgs');
    fx     = cppoptlib(x0, @rosenbrock);
    fx     = fminsearch(x0, @rosenbrock);

Even optimizing without any gradient information this library outperforms optimization routines from MATLAB on some problems.

# Benchmarks

Currently, not all solvers are equally good at all objective functions. The file `src/test/benchmark.cpp` contains some challenging objective functions which are tested by each provided solver. Note, MATLAB will also fail at some objective functions.

# Contribute

Make sure that `make lint` does not display any errors and check if travis is happy. Do not forget to `chmod +x lint.py`. It would be great, if some of the Forks-Owner are willing to make pull-request.

[eigen3]: http://eigen.tuxfamily.org/
[matlab]: http://www.mathworks.de/products/matlab/

