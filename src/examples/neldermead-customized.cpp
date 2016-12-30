#include <iostream>
#include <iomanip>
#include <fstream>

#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/neldermeadsolver.h"

static std::ofstream trace_stream;
static bool first_iteration = true; // to help generate trace JSON

// to use this library just use the namespace "cppoptlib"
namespace cppoptlib {

  // we define a new problem for optimizing the rosenbrock function
  // we use a templated-class rather than "auto"-lambda function for a clean architecture
  template<typename T> class Rosenbrock : public Problem<T, 2> {
    public:
      using typename Problem<T, 2>::TVector;
      using typename Problem<T, 2>::Scalar;
      using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

      // this is just the objective (NOT optional)
      T value(const TVector &x) {
        const T t1 = (1 - x[0]);
        const T t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
      }

      // bool callback(const cppoptlib::Criteria<T> &state, const TVector &x) {
      //   TVector a = x.transpose();

      //   // Be mindful of calls to value() in the callback if the function is
      //   // computationally intensive.expensive. Consider using detailed_callback().
      //   std::cout << std::setw(6) << state.iterations << std::setw(12) << a[0] << std::setw(12) << a[1] << std::setw(12) << value(x) << std::endl;
      //   return true;
      // }

      bool detailed_callback(const cppoptlib::Criteria<T> &state, SimplexOp op, int index, const MatrixType &x, std::vector<Scalar> f) {
        TVector xp = x.col(index).transpose();

        std::cout << std::setw(6) << state.iterations << std::setw(12) << xp[0] << std::setw(12) << xp[1] << std::setw(12) << f[index] << std::endl;

        // Write simplex trace
        TVector x0 = x.col(0).transpose();
        TVector x1 = x.col(1).transpose();
        TVector x2 = x.col(2).transpose();
        trace_stream << (first_iteration ? "" : ",\n") <<
          "    {\n"
          "      \"iter\": " << state.iterations << ",\n"
          "      \"op\": \"" << op << "\",\n"
          "      \"index\": " << index << ",\n"
          "      \"x\": [\n"
          "        [" << x0[0] << ", " << x0[1] << "],\n"
          "        [" << x1[0] << ", " << x1[1] << "],\n"
          "        [" << x2[0] << ", " << x2[1] << "]\n"
          "      ],\n"
          "      \"f\": [" << f[0] << ", " << f[1] << ", " << f[2] << "],\n"
          "      \"xDelta\": " << state.xDelta << ",\n"
          "      \"fDelta\": " << state.fDelta << "\n"
          "    }";
        first_iteration = false;

        return true;
      }

  };

  template <typename ProblemType> class ModifiedNelderMeadSolver: public NelderMeadSolver<ProblemType> {
    public:
      using Superclass = ISolver<ProblemType, 0>;
      using typename Superclass::Scalar;
      using typename Superclass::TVector;
      using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

      MatrixType makeInitialSimplex(TVector &x) {
        size_t dim = x.rows();

        // create initial simplex
        MatrixType s = MatrixType::Zero(dim, dim + 1);
        for (int c = 0; c < int(dim) + 1; ++c) {
          for (int r = 0; r < int(dim); ++r) {
            s(r, c) = x(r);
            if (r == c - 1) {
              s(r, c) = x(r) + 0.5;
            }
          }
        }

        NelderMeadSolver<ProblemType>::initialSimplexCreated = true;
        return s;
      }
  };
}

int main( int argc, char** argv ) {
  typedef double T;
  typedef cppoptlib::Rosenbrock<T> Rosenbrock;

  // Initialize the Rosenbrock-problem
  Rosenbrock f;

  // Choose a starting point
  Rosenbrock::TVector x(2); x << -1, 2;

  // Choose a solver
  cppoptlib::ModifiedNelderMeadSolver<Rosenbrock> solver;

  // Create a Criteria class to set the solver's stop conditions
  Rosenbrock::TCriteria crit = Rosenbrock::TCriteria::defaults();
  crit.iterations = 100;
  crit.fDelta = 0.0005;
  solver.setStopCriteria(crit);

  // Custom method defined in ModifiedNelderMeadSolver above
  solver.x0 = solver.makeInitialSimplex(x);

  // Write simplex trace (most of it will be written by the callback)
  trace_stream.open("simplex-trace.json", std::ofstream::out);
  trace_stream <<
    "{\n"
    "  \"simplex\": [\n";

  // and minimize the function
  solver.minimize(f, x);

  // Close trace output
  trace_stream <<
    "\n"
    "  ],\n"
    "  \"stop\": \"" << solver.stop_condition << "\"\n"
    "}\n";
  trace_stream.close();

  // Print results
  std::cout << std::string(42, '-') << std::endl;
  std::cout << "   stop:" << "  " << solver.stop_condition << std::endl;
  std::cout << "   argmin: " << x.transpose() << std::endl;
  std::cout << "   f in argmin: " << f(x) << std::endl;
  std::cout << "   trace written to simplex-trace.json" << std::endl;

  return 0;
}
