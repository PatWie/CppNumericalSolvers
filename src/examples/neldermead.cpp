#include <iostream>
#include <iomanip>

#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/neldermeadsolver.h"

// to use this library just use the namespace "cppoptlib"
namespace cppoptlib {

  // we define a new problem for optimizing the rosenbrock function
  // we use a templated-class rather than "auto"-lambda function for a clean architecture
  template<typename T> class Rosenbrock : public Problem<T, 2> {
    public:
      using typename Problem<T, 2>::TVector;

      // this is just the objective (NOT optional)
      T value(const TVector &x) {
        const T t1 = (1 - x[0]);
        const T t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
      }

      bool callback(const cppoptlib::Criteria<T> &state, const TVector &x) {
        TVector a = x.transpose();

        // Be mindful of calls to value() in the callback if the function is
        // computationally intensive.expensive. Consider using detailed_callback().
        std::cout << setw(6) << state.iterations << setw(12) << a[0] << setw(12) << a[1] << setw(12) << value(x) << std::endl;
        return true;
      }
  };
}

/**
 * @function main
 */
int main( int argc, char** argv ) {
  typedef double T;
  typedef cppoptlib::Rosenbrock<T> Rosenbrock;

  // initialize the Rosenbrock-problem
  Rosenbrock f;

  // choose a starting point
  Rosenbrock::TVector x(2); x << -1, 2;

  // choose a solver
  cppoptlib::NelderMeadSolver<Rosenbrock> solver;

  // and minimize the function
  solver.minimize(f, x);

  // print argmin
  std::cout << string(42, '-') << std::endl;
  std::cout << "   argmin: " << x.transpose() << std::endl;
  std::cout << "   f in argmin: " << f(x) << std::endl;

  return 0;
}
