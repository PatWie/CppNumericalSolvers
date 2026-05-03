// Copyright 2026, https://github.com/PatWie/CppNumericalSolvers
//
// Soft-margin SVM on Iris (versicolor vs. virginica), solved in the
// **dual** formulation with box constraints handled natively by
// L-BFGS-B.  The true Wolfe dual carries both box constraints and a
// single linear equality
//
//     sum_i alpha_i y_i = 0 ,
//
// which would normally require an outer augmented-Lagrangian wrapper
// around L-BFGS-B.  This example takes the common shortcut of
// dropping the equality (equivalent to fixing the intercept `b = 0`).
// On z-score-centred features the mean of each class is already
// symmetric about the origin, so the `b = 0` solution is very close
// to the full KKT optimum.  The sibling example `svm_dual_al.cc`
// restores the equality through the AL outer loop.
//
// Dual formulation (minimisation form):
//     min_{alpha}  0.5 * alpha^T Q alpha - 1^T alpha
//     subject to    0 <= alpha_i <= C
// with Q_ij = y_i y_j (x_i . x_j).  The primal weights are recovered
// after the solve as
//     w = sum_i alpha_i y_i x_i ,    b = 0 .

#include <iostream>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/lbfgsb.h"
#include "src/examples/iris_data.h"

namespace {

// Dual SVM objective on the alpha vector.  Stores the precomputed
// kernel-with-labels matrix `Q = (y*y^T) hadamard (X X^T)` so that
// each gradient evaluation is a single matrix-vector product.
class SvmDualObjective
    : public cppoptlib::function::FunctionCRTP<
          SvmDualObjective, double,
          cppoptlib::function::DifferentiabilityMode::First> {
 public:
  Eigen::MatrixXd kernel_matrix;  // Q, size n x n

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

}  // namespace

int main() {
  const auto data = cppoptlib::examples::LoadIrisVersicolorVirginica();
  const Eigen::MatrixXd& features = data.features;
  const Eigen::VectorXd& labels = data.labels;
  const Eigen::Index sample_count = features.rows();

  constexpr double regularisation_c = 1.0;

  SvmDualObjective dual_objective(features, labels);

  // Box bounds: 0 <= alpha_i <= C.
  Eigen::VectorXd lower_bound = Eigen::VectorXd::Zero(sample_count);
  Eigen::VectorXd upper_bound =
      Eigen::VectorXd::Constant(sample_count, regularisation_c);

  cppoptlib::solver::Lbfgsb<SvmDualObjective> solver;
  solver.SetBounds(lower_bound, upper_bound);

  // Start at the origin of alpha -- any feasible point works; the
  // origin lies at the corner of the box and trivially satisfies the
  // bounds.
  Eigen::VectorXd initial_alpha = Eigen::VectorXd::Zero(sample_count);
  auto [solution, progress] = solver.Minimize(
      dual_objective, cppoptlib::function::FunctionState(initial_alpha));

  // Recover primal weights.  `b = 0` by the intercept-dropped
  // simplification noted in the file header.
  const Eigen::VectorXd alpha = solution.x;
  const Eigen::VectorXd weighted_alphas = alpha.array() * labels.array();
  const Eigen::VectorXd w = features.transpose() * weighted_alphas;
  const double b = 0.0;

  const Eigen::VectorXd scores = (features * w).array() + b;
  const double accuracy =
      cppoptlib::examples::ClassificationAccuracy(scores, labels);

  // Count support vectors: alpha_i significantly above 0.  The
  // threshold `1e-5` matches the default constraint_threshold in this
  // library's stopping progress.
  constexpr double support_vector_threshold = 1e-5;
  const Eigen::Index support_vector_count =
      (alpha.array() > support_vector_threshold).count();

  std::cout << "SVM dual (L-BFGS-B, b = 0 relaxation)\n";
  std::cout << "  solver status:  " << progress.status << "\n";
  std::cout << "  iterations:     " << progress.num_iterations << "\n";
  std::cout << "  dual objective: " << solution.value << "\n";
  std::cout << "  support vecs:   " << support_vector_count << " / "
            << sample_count << "\n";
  std::cout << "  sum(alpha*y):   " << weighted_alphas.sum()
            << "   (would be 0 under the equality constraint)\n";
  std::cout << "  w:              " << w.transpose() << "\n";
  std::cout << "  b:              " << b << "\n";
  std::cout << "  accuracy:       " << accuracy << "\n";
  return 0;
}
