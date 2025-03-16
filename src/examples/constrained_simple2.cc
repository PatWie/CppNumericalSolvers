// Copyright 2025, https://github.com/PatWie/CppNumericalSolvers
#include <iostream>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/lbfgs.h"
//
// SumObjective: f(x) = x[0] + x[1] (to be minimized)
//
class SumObjective : public cppoptlib::function::FunctionXd<SumObjective> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Return the sum of the components.
  ScalarType operator()(const VectorType &x,
                        VectorType *gradient = nullptr) const {
    if (gradient) {
      // The gradient is a vector of ones.
      *gradient = VectorType::Ones(2);
    }
    return x.sum();
  }
};

//
// Circle: c(x) = x[0]^2 + x[1]^2
// This function will be used to form both an equality constraint (forcing
// the solution to lie on the circle) and an inequality constraint (ensuring
// the solution remains within the circle).
//
class Circle : public cppoptlib::function::FunctionXd<Circle> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Compute the squared norm.
  ScalarType operator()(const VectorType &x,
                        VectorType *gradient = nullptr) const {
    if (gradient) {
      *gradient = 2 * x;
    }
    return x.squaredNorm();
  }
};

//
// Main demo: solve the constrained problem
//
int main() {
  // Initial guess for x.
  SumObjective::VectorType x(2);
  x << 2, 10;

  // Define the objective: f(x) = x[0] + x[1].
  cppoptlib::function::FunctionExpr objective = SumObjective();

  // Define the circle function: c(x) = x[0]^2 + x[1]^2.
  cppoptlib::function::FunctionExpr circle = Circle();

  // Build the constrained optimization problem.
  // We impose two constraints:
  // 1. Equality constraint: circle(x) - 2 == 0, forcing the solution onto the
  // circle's boundary.
  // 2. Inequality constraint: 2 - circle(x) >= 0, ensuring the solution remains
  // inside the circle.
  cppoptlib::function::ConstrainedOptimizationProblem prob(
      objective,
      /* equality constraints */
      {cppoptlib::function::FunctionExpr(circle - 2)},
      /* inequality constraints */
      {cppoptlib::function::FunctionExpr(2 - circle)});

  // Set up an inner solver (LBFGS) for the unconstrained subproblems.
  cppoptlib::solver::Lbfgs<cppoptlib::function::FunctionExprXd> inner_solver;

  // Create the augmented Lagrangian solver using the inner solver.
  cppoptlib::solver::AugmentedLagrangian solver(prob, inner_solver);

  // Initialize the augmented Lagrange state.
  cppoptlib::solver::AugmentedLagrangeState<double> l_state(x, 1, 1, 1.0);

  // Run the solver.
  auto [solution, solver_state] = solver.Minimize(prob, l_state);

  // Output the results.
  std::cout << "Optimal f(x): " << objective(solution.x) << std::endl;
  std::cout << "Optimal x: " << solution.x.transpose() << std::endl;
  std::cout << "Iterations: " << solver_state.num_iterations << std::endl;
  std::cout << "Solver status: " << solver_state.status << std::endl;

  return 0;
}
