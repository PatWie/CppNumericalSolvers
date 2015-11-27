#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/bfgssolver.h"
#include "../../include/cppoptlib/solver/conjugatedgradientdescentsolver.h"
#include "../../include/cppoptlib/solver/newtondescentsolver.h"
#include "../../include/cppoptlib/solver/neldermeadsolver.h"
#include "../../include/cppoptlib/solver/lbfgssolver.h"
#include "../../include/cppoptlib/solver/cmaessolver.h"

// to use this library just use the namespace "cppoptlib"
namespace cppoptlib {

// we define a new problem for optimizing the rosenbrock function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template<typename T>
class Rosenbrock : public Problem<T> {
  public:
    // this is just the objective (NOT optional)
    T value(const Vector<T> &x) {
        const T t1 = (1 - x[0]);
        const T t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    }

    // if you calculated the derivative by hand
    // you can implement it here (OPTIONAL)
    // otherwise it will fall back to (bad) numerical finite differences
    void gradient(const Vector<T> &x, Vector<T> &grad) {
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1]  =                   200 * (x[1] - x[0] * x[0]);
    }

    // same for hessian (OPTIONAL)
    // if you want ot use 2nd-order solvers, I encourage you to specify the hessian
    // finite differences usually (this implementation) behave bad
    void hessian(const Vector<T> &x, Matrix<T> & hessian) {
        hessian(0, 0) = 1200 * x[0] * x[0] - 400 * x[1] + 1;
        hessian(0, 1) = -400 * x[0];
        hessian(1, 0) = -400 * x[0];
        hessian(1, 1) = 200;
    }

};

}
int main(int argc, char const *argv[]) {
    typedef float T;
    // initialize the Rosenbrock-problem
    cppoptlib::Rosenbrock<T> f;
    // choose a starting point
    cppoptlib::Vector<T> x(2); x << -1, 2;

    // first check the given derivative 
    // there is output, if they are NOT similar to finite differences
    bool probably_correct = f.checkGradient(x);

    // choose a solver
    //cppoptlib::BfgsSolver<T> solver;
    //cppoptlib::ConjugatedGradientDescentSolver<T> solver;
    //cppoptlib::NewtonDescentSolver<T> solver;
    //cppoptlib::NelderMeadSolver<T> solver;
    cppoptlib::LbfgsSolver<T> solver;
    //cppoptlib::CMAesSolver<T> solver;
    // and minimize the function
    solver.minimize(f, x);
    // print argmin
    std::cout << "argmin      " << x.transpose() << std::endl;
    std::cout << "f in argmin " << f(x) << std::endl;

    return 0;
}
