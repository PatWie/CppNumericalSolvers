// Copyright 2026, https://github.com/PatWie/CppNumericalSolvers
//
// Soft-margin SVM on Iris (versicolor vs. virginica), solved in the
// **full Wolfe dual** with both box constraints and the classifier
// equality constraint enforced properly.  The outer augmented-
// Lagrangian loop handles the equality; the inner L-BFGS-B handles
// the box constraints natively.  This combination -- AL outside,
// L-BFGS-B inside -- is the pattern you would use whenever a problem
// has box constraints that the inner solver can accept natively and
// also carries one or more non-box constraints that the inner solver
// cannot.
//
// Dual formulation (minimisation form):
//     min_{alpha}  0.5 * alpha^T Q alpha - 1^T alpha
//     subject to    0 <= alpha_i <= C
//                   sum_i alpha_i y_i = 0
// with Q_ij = y_i y_j (x_i . x_j).  The primal weights are recovered
// as
//     w = sum_i alpha_i y_i x_i ,    b = 0 .
// (The dual does not directly produce `b`; see `svm_dual_lbfgsb.cc`
// for the same simplification.  With the equality constraint now
// active in the dual, `w` still recovers the primal classifier up to
// this intercept.)

#include <iostream>
#include <vector>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/lbfgsb.h"
#include "src/examples/iris_data.h"

namespace {

// Dual SVM objective -- same as in `svm_dual_lbfgsb.cc` but restated
// here so the example stands alone.  Precomputes `Q = (y y^T) o (X X^T)`.
class SvmDualObjective
    : public cppoptlib::function::FunctionCRTP<
          SvmDualObjective, double,
          cppoptlib::function::DifferentiabilityMode::First> {
 public:
  Eigen::MatrixXd kernel_matrix;

  SvmDualObjective(const Eigen::MatrixXd& features,
                   const Eigen::VectorXd& labels) {
    const Eigen::MatrixXd gram = features * features.transpose();
    kernel_matrix = gram.array() * (labels * labels.transpose()).array();
  }

  ScalarType operator()(const VectorType& alpha,
                        VectorType* grad = nullptr) const {
    const Eigen::VectorXd q_alpha = kernel_matrix * alpha;
    const double value = 0.5 * alpha.dot(q_alpha) - alpha.sum();
    if (grad) {
      *grad = q_alpha - Eigen::VectorXd::Ones(alpha.size());
    }
    return value;
  }
};

// Equality constraint `c(alpha) = sum_i alpha_i y_i = 0`.
// Gradient is the constant label vector `y`.
class SvmDualEqualityConstraint
    : public cppoptlib::function::FunctionCRTP<
          SvmDualEqualityConstraint, double,
          cppoptlib::function::DifferentiabilityMode::First> {
 public:
  Eigen::VectorXd labels;

  explicit SvmDualEqualityConstraint(const Eigen::VectorXd& labels)
      : labels(labels) {}

  ScalarType operator()(const VectorType& alpha,
                        VectorType* grad = nullptr) const {
    if (grad) {
      *grad = labels;
    }
    return alpha.dot(labels);
  }
};

}  // namespace

int main() {
  const auto data = cppoptlib::examples::LoadIrisVersicolorVirginica();
  const Eigen::MatrixXd& features = data.features;
  const Eigen::VectorXd& labels = data.labels;
  const Eigen::Index sample_count = features.rows();

  constexpr double regularisation_c = 1.0;

  using FunctionExprD = cppoptlib::function::FunctionExpr<
      double, cppoptlib::function::DifferentiabilityMode::First>;
  FunctionExprD objective = SvmDualObjective(features, labels);
  FunctionExprD equality = SvmDualEqualityConstraint(labels);

  cppoptlib::function::ConstrainedOptimizationProblem problem(
      objective, /*eq=*/{equality});

  // Inner solver: L-BFGS-B with box bounds on alpha.  Configure once
  // here; the AugmentedLagrangian stores a copy of this solver and
  // reuses it for every outer iteration.  Copying an `Lbfgsb` copies
  // its `lower_bound_`, `upper_bound_`, and `bounds_initialized_`
  // fields, so the bounds survive into the outer loop.
  cppoptlib::solver::Lbfgsb<FunctionExprD> inner_solver;
  const Eigen::VectorXd lower_bound = Eigen::VectorXd::Zero(sample_count);
  const Eigen::VectorXd upper_bound =
      Eigen::VectorXd::Constant(sample_count, regularisation_c);
  inner_solver.SetBounds(lower_bound, upper_bound);

  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  // Start at alpha = 0 (corner of the box, trivially feasible for the
  // box constraints, violates the equality by 0).  Since
  // sum(0 * y) = 0 already, the outer-loop initial feasibility is
  // perfect; the first iteration's violation comes from wherever the
  // inner L-BFGS-B lands after one minimisation.
  Eigen::VectorXd initial_alpha = Eigen::VectorXd::Zero(sample_count);
  cppoptlib::solver::AugmentedLagrangeState<double> state(
      initial_alpha, /*num_eq=*/1, /*num_ineq=*/0, /*penalty=*/1.0);
  auto [solution, progress] = solver.Minimize(state);

  const Eigen::VectorXd alpha = solution.x;
  const Eigen::VectorXd weighted_alphas = alpha.array() * labels.array();
  const Eigen::VectorXd w = features.transpose() * weighted_alphas;
  const double b = 0.0;

  const Eigen::VectorXd scores = (features * w).array() + b;
  const double accuracy =
      cppoptlib::examples::ClassificationAccuracy(scores, labels);

  constexpr double support_vector_threshold = 1e-5;
  const Eigen::Index support_vector_count =
      (alpha.array() > support_vector_threshold).count();

  std::cout << "SVM dual (augmented Lagrangian + L-BFGS-B)\n";
  std::cout << "  solver status:     " << progress.status << "\n";
  std::cout << "  outer iterations:  " << progress.num_iterations << "\n";
  std::cout << "  max violation:     " << solution.max_violation << "\n";
  std::cout << "  multiplier lambda: "
            << solution.multiplier_state.equality_multipliers[0] << "\n";
  std::cout << "  support vectors:   " << support_vector_count << " / "
            << sample_count << "\n";
  std::cout << "  sum(alpha * y):    " << weighted_alphas.sum() << "\n";
  std::cout << "  w:                 " << w.transpose() << "\n";
  std::cout << "  b:                 " << b << "\n";
  std::cout << "  accuracy:          " << accuracy << "\n";
  return 0;
}
