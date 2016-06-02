#include <iostream>
#include <iomanip>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/gradientdescentsolver.h"

// we define a new problem for optimizing the Simple function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template<typename T>
class Simple : public cppoptlib::Problem<T, 2> {
  public:
    using typename cppoptlib::Problem<T, 2>::TVector; // Inherit the Vector typedef

    // this is just the objective (NOT optional)
    T value(const TVector &x) {
        return 5*x[0]*x[0] + 100*x[1]*x[1]+5;
    }

    // if you calculated the derivative by hand
    // you can implement it here (OPTIONAL)
    // otherwise it will fall back to (bad) numerical finite differences
    void gradient(const TVector &x, TVector &grad) {
        grad[0]  = 2*5*x[0];
        grad[1]  = 2*100*x[1];
    }

    bool callback(const cppoptlib::Criteria<T> &state, const TVector &x) {
        std::cout << "(" << std::setw(2) << state.iterations << ")"
                  << " ||dx|| = " << std::fixed << std::setw(8) << std::setprecision(4) << state.gradNorm
                  << " ||x|| = "  << std::setw(6) << x.norm()
                  << " f(x) = "   << std::setw(8) << value(x)
                  << " x = [" << std::setprecision(8) << x.transpose() << "]" << std::endl;
        return true;
    }
};

int main(int argc, char const *argv[]) {
    typedef Simple<double> TSimple;
    TSimple f;
    typename TSimple::TVector x; x << -10, 2;

    cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults(); // Create a Criteria class to set the solver's stop conditions
    crit.iterations = 10000;                              // Increase the number of allowed iterations
    cppoptlib::GradientDescentSolver<TSimple> solver;
    solver.setStopCriteria(crit);
    solver.minimize(f, x);
    std::cout << "f in argmin " << f(x) << std::endl;
    std::cout << "Solver status: " << solver.status() << std::endl;
    std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;
    return 0;
}
