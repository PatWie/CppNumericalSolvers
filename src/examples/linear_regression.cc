// Copyright 2025, https://github.com/PatWie/CppNumericalSolvers

#include <iostream>
#include <limits>

#include "Eigen/Core"
#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/lbfgs.h"
#include "cppoptlib/solver/lbfgsb.h"

using namespace cppoptlib::function;

class LinearRegression : public FunctionXf<LinearRegression> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScalarType operator()(const VectorType &x,
                        VectorType *gradient = nullptr) const {
    // Compute residuals for the two observations:
    // Observation 1: beta1 + 2*beta2 - 4
    // Observation 2: 3*beta1 + beta2 - 5
    ScalarType r1 = x[0] + 2 * x[1] - 4;
    ScalarType r2 = 3 * x[0] + x[1] - 5;

    // Compute the objective function: sum of squared errors
    ScalarType f = r1 * r1 + r2 * r2;

    // Compute the gradient if requested.
    // The gradient is: grad f = 2 * [ r1 + 3*r2, 2*r1 + r2 ]
    if (gradient) {
      gradient->resize(x.size());
      (*gradient)[0] = 2 * (r1 + 3 * r2);
      (*gradient)[1] = 2 * (2 * r1 + r2);
    }

    return f;
  }
};

class BoundConstraint : public FunctionXf<BoundConstraint> {
 public:
  int index;  // 0 or 1
  ScalarType lower_bound;

  BoundConstraint(int i, ScalarType bound) : index(i), lower_bound(bound) {}

  ScalarType operator()(const VectorType &x, VectorType *grad = nullptr) const {
    if (grad) {
      grad->setZero();
      (*grad)[index] = 1.0;  // ∇(x[i] - lower_bound)
    }
    return x[index] - lower_bound;
  }
};

int main() {
  cppoptlib::solver::Lbfgsb<FunctionExprXf> solver;

  // optimal solution is suppose to be [1, 1.6] under the box constraints
  FunctionExpr f = LinearRegression();

  // Either use L-BFG-B
  // ---------------------------
  Eigen::VectorXf x(2);
  x << -1, 2;
  Eigen::VectorXf lb(2);
  lb << 0, 1;
  Eigen::VectorXf ub(2);
  ub << 1, 2;
  solver.SetBounds(lb, ub);

  const auto initial_state = cppoptlib::function::FunctionState(x);
  auto [solution, solver_state] = solver.Minimize(f, initial_state);
  std::cout << "argmin " << solution.x.transpose() << std::endl;

  // Or model it as a augmented Lagrangian
  // ------------------------------------------
  FunctionExpr lb0 = BoundConstraint(0, 0.0f);
  FunctionExpr lb1 = BoundConstraint(1, 1.0f);
  FunctionExpr ub0 = -1 * BoundConstraint(0, 1.0f);
  FunctionExpr ub1 = -1 * BoundConstraint(1, 2.0f);

  ConstrainedOptimizationProblem prob(f,
                                      /* equality constraints */ {},
                                      /* inequality constraints */
                                      {
                                          lb0, /* x[0] - 0 >= 0 */
                                          lb1, /* x[1] - 1 >= 0 */
                                          ub0, /* 1 - x[0] >= 0 */
                                          ub1, /* 2 - x[1] >= 0 */
                                      });
  cppoptlib::solver::Lbfgs<FunctionExprXf> unconstrained_solver;
  cppoptlib::solver::AugmentedLagrangian aug_solver(prob, unconstrained_solver);
  cppoptlib::solver::AugmentedLagrangeState l_state(x, 0, 4, 1.0f);

  // Run the agumented solver.
  auto [aug_solution, aug_solver_state] = aug_solver.Minimize(prob, l_state);
  std::cout << "argmin " << aug_solution.x.transpose() << std::endl;

  return 0;
}
