// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <vector>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/bfgs.h"
#include "cppoptlib/solver/conjugated_gradient_descent.h"
#include "cppoptlib/solver/gradient_descent.h"
#include "cppoptlib/solver/lbfgs.h"
#include "cppoptlib/solver/lbfgsb.h"
#include "cppoptlib/solver/newton_descent.h"

using FunctionXd = cppoptlib::function::Function<double>;

// https://en.wikipedia.org/wiki/Rosenbrock_function
class Rosenbrock : public FunctionXd {
 public:
  using FunctionXd::hessian_t;
  using FunctionXd::vector_t;

  scalar_t operator()(const vector_t& x) const override {
    const double t1 = (1 - x[0]);
    const double t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  }
};

template <typename Solver>
void solve(const std::vector<double>& values) {
  Rosenbrock f;
  Rosenbrock::vector_t x(2);
  x << values[0], values[1];

  auto state = f.Eval(x);
  std::cout << "this"
            << "\n";

  std::cout << f(x) << "\n";
  std::cout << state.gradient << "\n";
  std::cout << state.hessian << "\n";

  std::cout << cppoptlib::utils::IsGradientCorrect(f, x) << "\n";
  std::cout << cppoptlib::utils::IsHessianCorrect(f, x) << "\n";

  Solver solver;

  auto [solution, solver_state] = solver.Minimize(f, x);
  std::cout << "argmin " << solution.x.transpose() << "\n";
  std::cout << "f in argmin " << solution.value << "\n";
  std::cout << "iterations " << solver_state.num_iterations << "\n";
  std::cout << "status " << solver_state.status << "\n";

  std::cout << "Solution: " << solution.x.transpose() << "\n";
  std::cout << "f(x): " << f(solution.x) << "\n";
}

int main(int argc, char* argv[]) {
  if (argc < 2) return EXIT_FAILURE;
  int solver_id = std::atoi(argv[1]);
  const std::vector<double> x{-100, 4};

  switch (solver_id) {
    case 0: {
      std::cout << "NewtonDescent\n";
      using Solver = cppoptlib::solver::NewtonDescent<Rosenbrock>;
      solve<Solver>(x);
      break;
    }
    case 1: {
      std::cout << "GradientDescent\n";
      using Solver = cppoptlib::solver::GradientDescent<Rosenbrock>;
      solve<Solver>(x);
      break;
    }

    case 2: {
      std::cout << "ConjugatedGradientDescent\n";
      using Solver = cppoptlib::solver::ConjugatedGradientDescent<Rosenbrock>;
      solve<Solver>(x);
      break;
    }

    case 3: {
      std::cout << "Bfgs\n";
      using Solver = cppoptlib::solver::Bfgs<Rosenbrock>;
      solve<Solver>(x);
      break;
    }

    case 4: {
      std::cout << "Lbfgs\n";
      using Solver = cppoptlib::solver::Lbfgs<Rosenbrock>;
      solve<Solver>(x);
      break;
    }

    case 5: {
      std::cout << "Lbfgsb\n";
      using Solver = cppoptlib::solver::Lbfgsb<Rosenbrock>;
      solve<Solver>(x);
      break;
    }
    default:
      std::cerr << "Invalid solver id\n";
      break;
  }
  return 0;
}
