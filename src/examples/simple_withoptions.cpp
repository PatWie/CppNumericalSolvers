#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/gradientdescentsolver.h"

// nolintnextline
using namespace cppoptlib;

// we define a new problem for optimizing the Simple function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template<typename T>
class Simple : public Problem<T> {
  public:
    // this is just the objective (NOT optional)
    T value(const Vector<T> &x) {
        return 5*x[0]*x[0] + 100*x[1]*x[1]+5;
    }

    // if you calculated the derivative by hand
    // you can implement it here (OPTIONAL)
    // otherwise it will fall back to (bad) numerical finite differences
    void gradient(const Vector<T> &x, Vector<T> &grad) {
        grad[0]  = 2*5*x[0];
        grad[1]  = 2*100*x[1];
    }

    bool callback(Criteria<T>* state, Vector<T> x) {
        printf("(%04d) ||dx||= %04.08f \t||x||= %02.08f \tf(x)= %04.010f \t", state->iterations, state->gradNorm, x.norm(), value(x));
        std::cout << " x = ["  <<  x.transpose() <<" ]" << std::endl;
        return true;
    }
};
int main(int argc, char const *argv[]) {

    Simple<double> f;
    Vector<double> x(2); x << -10, 2;

    Criteria<double> crit = Criteria<double>::defaults(); // Create a Criteria class to set the solver's stop conditions
    crit.iterations = 10000;                              // Increase the number of allowed iterations
    GradientDescentSolver<double> solver;
    solver.setStopCriteria(crit);
    solver.minimize(f, x);
    std::cout << "f in argmin " << f(x) << std::endl;
    std::cout << "Solver status: " << solver.status() << std::endl;
    std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;
    return 0;
}
