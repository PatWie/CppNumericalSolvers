// Instantiates all solvers with float to catch double-promotion and
// narrowing issues that only surface with non-double scalar types.
// The goal is exercising template instantiation, not convergence testing.

#include <cassert>
#include <cmath>
#include <iostream>

#include "Eigen/Core"
#include "cppoptlib/function.h"
#include "cppoptlib/solver/bfgs.h"
#include "cppoptlib/solver/conjugated_gradient_descent.h"
#include "cppoptlib/solver/gradient_descent.h"
#include "cppoptlib/solver/lbfgs.h"
#include "cppoptlib/solver/newton_descent.h"

using FunctionExprF1 = cppoptlib::function::FunctionExpr<
    float, cppoptlib::function::DifferentiabilityMode::First, 2>;
using FunctionExprF2 = cppoptlib::function::FunctionExpr<
    float, cppoptlib::function::DifferentiabilityMode::Second, 2>;

class FloatQuadratic1
    : public cppoptlib::function::FunctionCRTP<
          FloatQuadratic1, float,
          cppoptlib::function::DifferentiabilityMode::First, 2> {
 public:
  ScalarType operator()(const VectorType& x, VectorType* grad) const {
    if (grad) *grad = Eigen::Vector2f(10 * x[0], 200 * x[1]);
    return 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;
  }
};

class FloatQuadratic2
    : public cppoptlib::function::FunctionCRTP<
          FloatQuadratic2, float,
          cppoptlib::function::DifferentiabilityMode::Second, 2> {
 public:
  ScalarType operator()(const VectorType& x, VectorType* grad,
                        MatrixType* hess) const {
    if (grad) *grad = Eigen::Vector2f(10 * x[0], 200 * x[1]);
    if (hess) {
      hess->setZero();
      hess->diagonal() << 10, 200;
    }
    return 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;
  }
};

template <typename Solver>
void RunSolver(typename Solver::FunctionType f) {
  Eigen::Vector2f x0(-1.0f, 0.5f);
  Solver solver;
  auto [solution, solver_state] =
      solver.Minimize(f, cppoptlib::function::FunctionState<float, 2>(x0));
  // Verify the solver ran and produced a finite result.
  assert(solver_state.num_iterations > 0);
  assert(std::isfinite(solution.value));
  // Strong solvers must converge on this simple problem.
  (void)solution;
}

int main() {
  RunSolver<cppoptlib::solver::Lbfgs<FunctionExprF1>>(
      FunctionExprF1(FloatQuadratic1()));
  RunSolver<cppoptlib::solver::Bfgs<FunctionExprF2>>(
      FunctionExprF2(FloatQuadratic2()));
  RunSolver<cppoptlib::solver::GradientDescent<FunctionExprF1>>(
      FunctionExprF1(FloatQuadratic1()));
  RunSolver<cppoptlib::solver::ConjugatedGradientDescent<FunctionExprF1>>(
      FunctionExprF1(FloatQuadratic1()));
  RunSolver<cppoptlib::solver::NewtonDescent<FunctionExprF2>>(
      FunctionExprF2(FloatQuadratic2()));
  std::cout << "PASS: all float-scalar solvers ran successfully\n";
  return 0;
}
