#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/bfgssolver.h"

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
};
int main(int argc, char const *argv[]) {

    Simple<double> f;
    Vector<double> x(2); x << -1, 2;
    BfgsSolver<double> solver;
    solver.minimize(f, x);
    std::cout << "f in argmin " << f(x) << std::endl;
    return 0;
}
