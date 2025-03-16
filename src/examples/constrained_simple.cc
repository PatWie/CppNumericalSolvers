// Copyright 2025, https://github.com/PatWie/CppNumericalSolvers
#include <iostream>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/lbfgs.h"

// Define a 2D function type with first-order differentiability.
template <class F>
using Function2d = cppoptlib::function::FunctionCRTP<
    F, double, cppoptlib::function::DifferentiabilityMode::First, 2>;

// Alias for an "any function" with the above properties.
using FunctionExpr2d = cppoptlib::function::FunctionExpr<
    double, cppoptlib::function::DifferentiabilityMode::First, 2>;

//
// QuadraticObjective2: f(x) = (x[0]-1)^2 + (x[1]-2)^2
//
// The unconstrained optimum is (1,2) with f(x)=0.
// However, the constraints (below) force a different solution.
//
class QuadraticObjective2 : public Function2d<QuadraticObjective2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScalarType operator()(const VectorType &x,
                        VectorType *gradient = nullptr) const {
    if (gradient) {
      // Gradient: 2*(x - [1,2])
      VectorType ref(2);
      ref << 1, 2;
      *gradient = 2 * (x - ref);
    }
    return (x(0) - 1) * (x(0) - 1) + (x(1) - 2) * (x(1) - 2);
  }
};

//
// EqualityConstraint2: g(x) = x[0] - 0.5 = 0
//
// This forces x[0] to be exactly 0.5.
// Thus, the optimal solution must have x[0] = 0.5.
//
class EqualityConstraint2 : public Function2d<EqualityConstraint2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScalarType operator()(const VectorType &x,
                        VectorType *gradient = nullptr) const {
    if (gradient) {
      // Gradient is [1, 0].
      *gradient = (VectorType(2) << 1, 0).finished();
    }
    return x(0) - 0.5;
  }
};

//
// InequalityConstraint3: h(x) = 2 - (x[0] + x[1]) >= 0
//
// This requires x[0] + x[1] <= 2. With x[0]=0.5 from the equality constraint,
// we have x[1] <= 1.5.
//
class InequalityConstraint3 : public Function2d<InequalityConstraint3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScalarType operator()(const VectorType &x,
                        VectorType *gradient = nullptr) const {
    if (gradient) {
      // Gradient is [-1, -1].
      *gradient = (VectorType(2) << -1, -1).finished();
    }
    return 2 - (x(0) + x(1));
  }
};

//
// Main demo: Solve the constrained problem with multiple constraints
//
int main() {
  // Initial guess for x. We choose a starting point that is not optimal.
  QuadraticObjective2::VectorType x(2);
  x << 1, 1;

  // Define the objective function.
  cppoptlib::function::FunctionExpr objective = QuadraticObjective2();

  // Define the constraint functions.
  cppoptlib::function::FunctionExpr eq = EqualityConstraint2();
  cppoptlib::function::FunctionExpr ineq = InequalityConstraint3();

  // Construct the constrained optimization problem.
  //
  // Problem Statement:
  //   minimize f(x) = (x[0]-1)^2 + (x[1]-2)^2
  //   subject to:
  //     equality constraint:  x[0] - 0.5 == 0   (forces x[0]=0.5)
  //     inequality constraint: 2 - (x[0]+x[1]) >= 0  (forces x[0]+x[1] <= 2)
  //
  // Expected Outcome:
  //   The unconstrained optimum is (1,2) with f(x)=0, but it is infeasible
  //   because 1+2=3 > 2. With x[0] forced to 0.5, the inequality requires x[1]
  //   <= 1.5. The best feasible choice is x = (0.5, 1.5), giving:
  //       f(x) = (0.5-1)^2 + (1.5-2)^2 = 0.25 + 0.25 = 0.5.
  cppoptlib::function::ConstrainedOptimizationProblem prob(
      objective,
      /* equality constraints */ {eq},
      /* inequality constraints */ {ineq});

  // Set up an inner LBFGS solver for unconstrained subproblems.
  cppoptlib::solver::Lbfgs<FunctionExpr2d> unconstrained_solver;

  // Create the augmented Lagrangian solver that handles the constraints.
  cppoptlib::solver::AugmentedLagrangian solver(prob, unconstrained_solver);

  // Initialize the augmented Lagrange state.
  cppoptlib::solver::AugmentedLagrangeState<double, 2> l_state(x, 1, 1, 1.0);

  // Run the solver.
  auto [solution, solver_state] = solver.Minimize(prob, l_state);

  // Output the results.
  std::cout << "Optimal f(x): " << objective(solution.x) << std::endl;
  std::cout << "Optimal x: " << solution.x.transpose() << std::endl;
  std::cout << "Iterations: " << solver_state.num_iterations << std::endl;
  std::cout << "Solver status: " << solver_state.status << std::endl;

  // Expected Output:
  //   Optimal x should be close to (0.5, 1.5).
  //   The optimal function value f(x) should be approximately 0.5.
  //   Both constraints are active: x[0]=0.5 (equality) and x[0]+x[1]=2
  //   (inequality).
  return 0;
}
