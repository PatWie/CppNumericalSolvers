// Copyright 2025, https://github.com/PatWie/CppNumericalSolvers
#include <iostream>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/lbfgs.h"

// Define a 2D function type with first-order differentiability.
template <class F>
using Function2d = cppoptlib::function::FunctionCRTP<
    F, double, cppoptlib::function::DifferentiabilityMode::First, 2>;

// Define an alias for an "any function" with the above properties.
using AnyFunction2d1 = cppoptlib::function::AnyFunction<
    double, cppoptlib::function::DifferentiabilityMode::First, 2>;

//
// SumObjective: f(x) = x[0] + x[1] (to be minimized)
//
class SumObjective : public Function2d<SumObjective> {
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
class Circle : public Function2d<Circle> {
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
  cppoptlib::function::AnyFunction objective = SumObjective();

  // Define the circle function: c(x) = x[0]^2 + x[1]^2.
  cppoptlib::function::AnyFunction circle = Circle();

  // Build the constrained optimization problem.
  // We impose two constraints:
  // 1. Equality constraint: circle(x) - 2 == 0, forcing the solution onto the
  // circle's boundary.
  // 2. Inequality constraint: 2 - circle(x) >= 0, ensuring the solution remains
  // inside the circle.
  cppoptlib::function::ConstrainedOptimizationProblem prob(
      objective,
      /* equality constraints */ {circle - 2},
      /* inequality constraints */ {2 - circle});

  // Set up an inner solver (LBFGS) for the unconstrained subproblems.
  cppoptlib::solver::Lbfgs<AnyFunction2d1> inner_solver;

  // Create the augmented Lagrangian solver using the inner solver.
  cppoptlib::solver::AugmentedLagrangian<decltype(prob), decltype(inner_solver)>
      solver(inner_solver);

  // Initialize the augmented Lagrange state.
  cppoptlib::solver::AugmentedLagrangeState<double, 2> l_state;
  l_state.x = x; // set the initial guess
  // Initialize multipliers: one for the equality and one for the inequality
  // constraint.
  l_state.multiplier_state =
      cppoptlib::function::LagrangeMultiplierState<double>({1}, {1});
  // Initialize the penalty parameter.
  l_state.penalty_state = cppoptlib::function::PenaltyState<double>(1);

  // Run the solver.
  auto [solution, solver_state] = solver.Minimize(prob, l_state);

  // Output the results.
  std::cout << "Optimal f(x): " << objective(solution.x) << std::endl;
  std::cout << "Optimal x: " << solution.x.transpose() << std::endl;
  std::cout << "Iterations: " << solver_state.num_iterations << std::endl;
  std::cout << "Solver status: " << solver_state.status << std::endl;

  return 0;
}
