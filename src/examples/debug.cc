#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "cppoptlib/function.h"

using namespace cppoptlib::function;

// LinearFunction: supports first-order information.
struct LinearFunction : public FunctionXd<LinearFunction> {
  Eigen::VectorXd m;
  double b;

  LinearFunction(const Eigen::VectorXd &m_, double b_) : m(m_), b(b_) {}

  // Simplified operator(): only function value and gradient.
  double operator()(const VectorType &x, VectorType *grad = nullptr) const {
    double fx = m.dot(x) + b;
    if (grad) {
      *grad = m;
    }
    return fx;
  }
};

// ConstantFunction: supports no derivative information.
struct ConstantFunction
    : public cppoptlib::function::FunctionCRTP<ConstantFunction, double,
                                               DifferentiabilityMode::None> {
  double c;
  ConstantFunction(double c_) : c(c_) {}

  double operator()(const VectorType &x) const {
    (void)x;
    return c;
  }
};

// QuadraticFunction: supports second-order information.
struct QuadraticFunction
    : public cppoptlib::function::FunctionCRTP<QuadraticFunction, double,
                                               DifferentiabilityMode::Second> {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  double c;

  QuadraticFunction(const Eigen::MatrixXd &A_, const Eigen::VectorXd &b_,
                    double c_)
      : A(A_), b(b_), c(c_) {}

  double operator()(const VectorType &x, VectorType *grad = nullptr,
                    MatrixType *hess = nullptr) const {
    double fx = 0.5 * x.transpose() * A * x + b.dot(x) + c;
    if (grad) {
      *grad = A * x + b;
    }
    if (hess) {
      *hess = A;
    }
    return fx;
  }
};
struct QuadraticFunction2
    : public cppoptlib::function::FunctionCRTP<
          QuadraticFunction2, double, DifferentiabilityMode::Second, 2> {
  Eigen::Matrix2d A;
  Eigen::Vector2d b;
  double c;

  QuadraticFunction2(const Eigen::Matrix2d &A_, const Eigen::Vector2d &b_,
                     double c_)
      : A(A_), b(b_), c(c_) {}

  double operator()(const VectorType &x, VectorType *grad = nullptr,
                    MatrixType *hess = nullptr) const {
    double fx = 0.5 * x.transpose() * A * x + b.dot(x) + c;
    if (grad) {
      *grad = A * x + b;
    }
    if (hess) {
      *hess = A;
    }
    return fx;
  }
};

//-----------------------------------------------------------------
// Example Usage
int main() {
  using VectorType = Eigen::VectorXd;
  using MatrixType = Eigen::MatrixXd;

  // Test LinearFunction via FunctionExpr (First-order).
  {
    VectorType m(2);
    m << 2, -1;
    double b_val = 0.5;
    LinearFunction lin(m, b_val);
    FunctionExpr anyLin(lin);

    VectorType x(2);
    x << 1, 2;
    VectorType grad(2);
    double f_lin = anyLin(x, &grad);
    std::cout << "Linear function value: " << f_lin << "\n";
    std::cout << "Gradient: " << grad.transpose() << "\n\n";
  }

  // Test ConstantFunction via FunctionExpr (None).
  {
    ConstantFunction cf(3.14);
    FunctionExpr anyConst(cf);

    VectorType x(2);
    x << 1, 2;
    double f_const = anyConst(x);
    std::cout << "Constant function value: " << f_const << "\n\n";

    FunctionExpr negAnyConst = -anyConst;
    f_const = negAnyConst(x);
    std::cout << "Constant function value: " << f_const << "\n\n";
  }

  // Test QuadraticFunction via FunctionExpr (Second-order).
  {
    MatrixType A(2, 2);
    A << 3, 1, 1, 2;
    VectorType b(2);
    b << 1, -1;
    double c_val = 0.5;
    QuadraticFunction quad(A, b, c_val);
    FunctionExpr anyQuad(quad);

    VectorType x(2);
    x << 1, 2;
    VectorType grad(2);
    MatrixType hess(2, 2);
    double f_quad = anyQuad(x, &grad, &hess);
    std::cout << "Quadratic function value: " << f_quad << "\n";
    std::cout << "Gradient: " << grad.transpose() << "\n";
    std::cout << "Hessian:\n" << hess << "\n\n";
  }
  // Test QuadraticFunction2 via FunctionExpr (Second-order).
  {
    using VectorType = Eigen::Vector2d;
    using MatrixType = Eigen::Matrix2d;
    MatrixType A;
    A << 3, 1, 1, 2;
    VectorType b;
    b << 1, -1;
    double c_val = 0.5;
    QuadraticFunction2 quad(A, b, c_val);
    FunctionExpr anyQuad(quad);

    VectorType x(2);
    x << 1, 2;
    VectorType grad(2);
    MatrixType hess(2, 2);
    double f_quad = anyQuad(x, &grad, &hess);
    std::cout << "Quadratic function value: " << f_quad << "\n";
    std::cout << "Gradient: " << grad.transpose() << "\n";
    std::cout << "Hessian:\n" << hess << "\n\n";
  }

  // Test expression templates:
  // Expression: (quad + lin - lin * 2.0)
  {
    MatrixType A(2, 2);
    A << 3, 1, 1, 2;
    VectorType b(2);
    b << 1, -1;
    double c_val = 0.5;
    QuadraticFunction quad(A, b, c_val);

    VectorType m(2);
    m << 2, -1;
    double b_val = 0.5;
    LinearFunction lin(m, b_val);

    // Expression: quad + lin  (Min(Differentiability(quad),
    // Differentiability(lin)) = First)
    auto expr1 = quad + lin;
    // Expression: lin * 2.0
    auto expr2 = lin * 2.0;
    // Expression: (quad + lin) - (lin * 2.0)
    auto expr = expr1 - expr2;

    FunctionExpr anyExpr(expr);
    VectorType x(2);
    x << 1, 2;
    VectorType grad(2);
    double f_expr = anyExpr(x, &grad);
    std::cout << "Expression (quad + lin - lin*2.0) value: " << f_expr << "\n";
    std::cout << "Expression gradient: " << grad.transpose() << "\n";

    FunctionExpr prodFunc(quad * quad);
    VectorType grad2(2);
    quad(x, &grad2);
    std::cout << "Expression gradient: " << grad2.transpose() << "\n";
    double f2_expr = prodFunc(x, &grad2);
    std::cout << "Expression (quad * quad) value: " << f2_expr << "\n";
    std::cout << "Expression gradient: " << grad2.transpose() << "\n";
  }

  // --- Equality Constraint Example ---
  // Constraint: 2*x0 + x1 - 3 == 0
  // Penalty function: 0.5 * [2*x0 + x1 - 3]^2
  {
    // Define a linear function: f(x) = 2*x0 + 1*x1 - 3.
    VectorType m(2);
    m << 2, 1;
    double b_val = -3;
    FunctionExpr anyEqPenalty =
        quadraticEqualityPenalty(LinearFunction(m, b_val));

    // Evaluate at x = (1, 2)
    VectorType x(2);
    x << 1, 2;
    VectorType grad(2);
    double eqPenaltyValue = anyEqPenalty(x, &grad);
    std::cout << "Equality constraint penalty (2*x0 + x1 - 3 == 0) at x=(1,2): "
              << eqPenaltyValue << "\n";
    std::cout << "Gradient: " << grad.transpose() << "\n\n";
  }

  {
    using TScalar = double;
    using VectorType = Eigen::VectorXd;
    using MatrixType = Eigen::MatrixXd;

    // Define the objective function as a quadratic:
    MatrixType A(2, 2);
    A << 3, 1, 1, 2;
    VectorType b(2);
    b << 1, -1;
    TScalar c_val = 0.5;
    FunctionExpr objective = QuadraticFunction(A, b, c_val);

    // Define two constraints.
    // Constraint 0: Equality constraint: 2*x0 + x1 - 3 == 0.
    VectorType m1(2);
    m1 << 2, 1;
    TScalar b1 = -3;
    FunctionExpr eqConstraint = LinearFunction(m1, b1);

    // Constraint 1: Inequality constraint: x0 - 1 <= 0.
    // (We interpret "≤" as an InequalityLe constraint.)
    VectorType m2(2);
    m2 << 1, 0;
    TScalar b2 = -1;
    FunctionExpr ineqConstraint = LinearFunction(m2, b2);

    // Build the optimization problem.
    ConstrainedOptimizationProblem optimization_problem(
        objective, {FunctionExpr(-eqConstraint)}, {ineqConstraint});

    LagrangeMultiplierState<double> l_state(1, 1);
    l_state.equality_multipliers[0] = 1;
    l_state.inequality_multipliers[0] = 1;
    l_state = LagrangeMultiplierState<double>({1}, {1});
    PenaltyState<double> p_state(1);
    FunctionExpr unconstrainedPenalty =
        ToPenalty(optimization_problem, p_state);
    FunctionExpr unconstrainedAugmentedLagrangian =
        ToAugmentedLagrangian(optimization_problem, l_state, p_state);
    VectorType x(2);
    x << 1, 3;

    FunctionExpr penalty_part = FormPenaltyPart(optimization_problem, p_state);
    FunctionExpr l_part = FormLagrangianPart(optimization_problem, l_state);

    // Evaluate at the initial state.
    VectorType grad(2);
    TScalar objValue;
    objValue = eqConstraint(x, &grad);
    std::cout << "eqConstraint at x0: " << objValue << "\n";
    objValue = ineqConstraint(x, &grad);
    std::cout << "ineqConstraint at x0: " << objValue << "\n";
    objValue = unconstrainedAugmentedLagrangian(x, &grad);
    std::cout << "Augmented Lagrangian value at x0: " << objValue << "\n";
    objValue = l_part(x, &grad);
    std::cout << "Lpart value at x0: " << objValue << "\n";
    objValue = penalty_part(x, &grad);
    std::cout << "Penalty value at x0: " << objValue << "\n";
    objValue = unconstrainedPenalty(x, &grad);
    std::cout << "obj + penalty value at x0: " << objValue << "\n";
    std::cout << "Gradient: " << grad.transpose() << "\n";
  }

  return 0;
}
