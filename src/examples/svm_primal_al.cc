// Copyright 2026, https://github.com/PatWie/CppNumericalSolvers
//
// Soft-margin SVM on Iris (versicolor vs. virginica), solved via the
// **constrained** primal formulation using the augmented-Lagrangian
// solver.  The sibling example `svm_primal_lbfgs.cc` folds the margin
// constraint into the objective as a squared hinge; here we leave it
// as an explicit inequality and let the outer AL loop discover the
// Lagrange multipliers.
//
// Primal constrained formulation:
//     min_{w, b, xi}  0.5 * ||w||^2 + C * sum_i xi_i
//     subject to       y_i (w.x_i + b) - 1 + xi_i >= 0   (margin)
//                      xi_i >= 0                           (non-negativity)
//
// Variable layout in `x`: first `d` entries are `w`, entry `d` is the
// bias `b`, entries `d + 1 .. d + n` are the slacks `xi`.  Total
// dimension is `d + 1 + n`.

#include <iostream>
#include <vector>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/lbfgs.h"
#include "src/examples/iris_data.h"

namespace {

using cppoptlib::examples::iris_feature_count;
using cppoptlib::examples::iris_sample_count;

// Primal objective `f(w, b, xi) = 0.5 ||w||^2 + C * sum(xi)`.
class SvmPrimalObjective
    : public cppoptlib::function::FunctionCRTP<
          SvmPrimalObjective, double,
          cppoptlib::function::DifferentiabilityMode::First> {
 public:
  int feature_count;  // d
  int sample_count;   // n
  double regularisation_c;

  SvmPrimalObjective(int feature_count, int sample_count, double c)
      : feature_count(feature_count),
        sample_count(sample_count),
        regularisation_c(c) {}

  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    const Eigen::VectorXd w = x.head(feature_count);
    // Skip the bias (index `feature_count`); slacks start at
    // `feature_count + 1`.
    const Eigen::VectorXd xi = x.segment(feature_count + 1, sample_count);
    const double value = 0.5 * w.squaredNorm() + regularisation_c * xi.sum();
    if (grad) {
      *grad = VectorType::Zero(x.size());
      grad->head(feature_count) = w;
      // Bias has no gradient contribution from the objective.
      grad->segment(feature_count + 1, sample_count)
          .setConstant(regularisation_c);
    }
    return value;
  }
};

// Margin constraint for sample `i`:
//     c_i(w, b, xi) = y_i * (w . x_i + b) - 1 + xi_i  >=  0.
// Gradient: dc/dw = y_i * x_i, dc/db = y_i, dc/dxi_j = delta_{ij}.
class SvmMarginConstraint
    : public cppoptlib::function::FunctionCRTP<
          SvmMarginConstraint, double,
          cppoptlib::function::DifferentiabilityMode::First> {
 public:
  Eigen::VectorXd feature_row;  // x_i, dimension d
  double label;                 // y_i in {-1, +1}
  int sample_index;             // i, for the slack component.
  int feature_count;            // d
  int sample_count;             // n

  SvmMarginConstraint(const Eigen::VectorXd& feature_row, double label,
                      int sample_index, int feature_count, int sample_count)
      : feature_row(feature_row),
        label(label),
        sample_index(sample_index),
        feature_count(feature_count),
        sample_count(sample_count) {}

  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    const Eigen::VectorXd w = x.head(feature_count);
    const double b = x(feature_count);
    const double xi_i = x(feature_count + 1 + sample_index);
    const double constraint_value =
        label * (feature_row.dot(w) + b) - 1.0 + xi_i;
    if (grad) {
      *grad = VectorType::Zero(x.size());
      grad->head(feature_count) = label * feature_row;
      (*grad)(feature_count) = label;
      (*grad)(feature_count + 1 + sample_index) = 1.0;
    }
    return constraint_value;
  }
};

// Slack non-negativity constraint for sample `i`:
//     c_i(w, b, xi) = xi_i  >=  0.
// Gradient: delta on the `xi_i` entry.
class SvmSlackConstraint
    : public cppoptlib::function::FunctionCRTP<
          SvmSlackConstraint, double,
          cppoptlib::function::DifferentiabilityMode::First> {
 public:
  int sample_index;   // i
  int feature_count;  // d

  SvmSlackConstraint(int sample_index, int feature_count)
      : sample_index(sample_index), feature_count(feature_count) {}

  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    const double xi_i = x(feature_count + 1 + sample_index);
    if (grad) {
      *grad = VectorType::Zero(x.size());
      (*grad)(feature_count + 1 + sample_index) = 1.0;
    }
    return xi_i;
  }
};

}  // namespace

int main() {
  const auto data = cppoptlib::examples::LoadIrisVersicolorVirginica();
  const Eigen::MatrixXd& features = data.features;
  const Eigen::VectorXd& labels = data.labels;
  const int feature_count = static_cast<int>(features.cols());
  const int sample_count = static_cast<int>(features.rows());
  const int variable_count = feature_count + 1 + sample_count;

  constexpr double regularisation_c = 1.0;

  // Assemble the problem.  Objective is the sum of primal terms; the
  // constraint list is a flat vector we build up by value.
  using FunctionExprD = cppoptlib::function::FunctionExpr<
      double, cppoptlib::function::DifferentiabilityMode::First>;

  FunctionExprD objective =
      SvmPrimalObjective(feature_count, sample_count, regularisation_c);

  std::vector<FunctionExprD> inequality_constraints;
  inequality_constraints.reserve(2 * sample_count);
  for (int i = 0; i < sample_count; ++i) {
    const Eigen::VectorXd feature_row = features.row(i).transpose();
    inequality_constraints.emplace_back(SvmMarginConstraint(
        feature_row, labels(i), i, feature_count, sample_count));
  }
  for (int i = 0; i < sample_count; ++i) {
    inequality_constraints.emplace_back(SvmSlackConstraint(i, feature_count));
  }

  cppoptlib::function::ConstrainedOptimizationProblem problem(
      objective, /*eq=*/{}, inequality_constraints);

  // Inner solver: L-BFGS on the augmented composite.  The composite
  // dimension is `d + 1 + n = 105` for Iris.
  cppoptlib::solver::Lbfgs<FunctionExprD> inner_solver;

  cppoptlib::solver::AugmentedLagrangian<decltype(problem),
                                         decltype(inner_solver)>
      solver(problem, inner_solver);

  // Initial state: start at the origin with slacks at zero.  The
  // primal is infeasible at `xi = 0` because most samples will have
  // negative margins; the AL outer loop pulls the iterate onto the
  // feasible boundary.
  Eigen::VectorXd initial_x = Eigen::VectorXd::Zero(variable_count);
  // Use zero-initialised multipliers; initial penalty of 1.0 matches
  // the rest of the constrained examples in this repo.
  cppoptlib::solver::AugmentedLagrangeState<double> state(
      initial_x, /*num_eq=*/0,
      /*num_ineq=*/static_cast<size_t>(2 * sample_count),
      /*penalty=*/1.0);

  auto [solution, progress] = solver.Minimize(state);

  const Eigen::VectorXd w = solution.x.head(feature_count);
  const double b = solution.x(feature_count);
  const Eigen::VectorXd scores = (features * w).array() + b;
  const double accuracy =
      cppoptlib::examples::ClassificationAccuracy(scores, labels);

  std::cout << "SVM primal (augmented Lagrangian + L-BFGS)\n";
  std::cout << "  solver status:  " << progress.status << "\n";
  std::cout << "  outer iters:    " << progress.num_iterations << "\n";
  std::cout << "  max violation:  " << solution.max_violation << "\n";
  std::cout << "  objective:      "
            << 0.5 * w.squaredNorm() +
                   regularisation_c *
                       solution.x.segment(feature_count + 1, sample_count).sum()
            << "\n";
  std::cout << "  w:              " << w.transpose() << "\n";
  std::cout << "  b:              " << b << "\n";
  std::cout << "  accuracy:       " << accuracy << "\n";
  return 0;
}
