// Copyright 2026, https://github.com/PatWie/CppNumericalSolvers
//
// Soft-margin SVM on Iris (versicolor vs. virginica), solved by
// calling L-BFGS directly on the unconstrained penalised form of the
// primal problem.  This is the simplest of the four SVM examples and
// establishes the baseline classifier against which the others are
// cross-checked.
//
// Primal formulation:
//     min_{w, b}  0.5 * ||w||^2 + C * sum_i L(y_i (w.x_i + b))
// where L is the *squared* hinge loss
//     L(m) = max(0, 1 - m)^2 .
// Squared hinge gives a continuously differentiable objective (the
// plain hinge has a kink at `m = 1` that trips up gradient-based line
// searches) and the same optimal classifier up to a scale factor.
//
// Gradients:
//     dL/dm = -2 * max(0, 1 - m)
//     df/dw = w   - 2C * sum_i max(0, 1 - m_i) * y_i * x_i
//     df/db =     - 2C * sum_i max(0, 1 - m_i) * y_i .

#include <iostream>

#include "cppoptlib/function.h"
#include "cppoptlib/solver/lbfgs.h"
#include "src/examples/iris_data.h"

namespace {

using cppoptlib::examples::iris_feature_count;

// Soft-margin SVM primal objective: ||w||^2 / 2 + C * sum(squared hinge).
// Variable layout in `x`: the first `iris_feature_count` entries are
// `w`, the final entry is the bias `b`.
class SvmPrimalSquaredHinge
    : public cppoptlib::function::FunctionCRTP<
          SvmPrimalSquaredHinge, double,
          cppoptlib::function::DifferentiabilityMode::First> {
 public:
  // Reference to the training data and hyperparameter.  Stored by
  // reference to avoid a second copy -- the caller outlives the
  // objective.
  const Eigen::MatrixXd& features;  // n x d
  const Eigen::VectorXd& labels;    // n, values +/- 1
  double regularisation_c;

  SvmPrimalSquaredHinge(const Eigen::MatrixXd& features,
                        const Eigen::VectorXd& labels, double c)
      : features(features), labels(labels), regularisation_c(c) {}

  // Dimension at runtime: `d + 1` where `d` is the feature count.
  int GetDimension() const { return static_cast<int>(features.cols()) + 1; }

  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr) const {
    const Eigen::Index d = features.cols();
    const Eigen::VectorXd w = x.head(d);
    const double b = x(d);

    // Margin per sample: m_i = y_i (x_i . w + b).
    const Eigen::VectorXd scores = (features * w).array() + b;
    const Eigen::VectorXd margins = labels.array() * scores.array();

    // Squared hinge loss per sample: L_i = max(0, 1 - m_i)^2.
    const Eigen::VectorXd slacks =
        (1.0 - margins.array()).max(0.0).matrix();  // max(0, 1 - m_i).
    const double hinge_sum = slacks.squaredNorm();
    const double value = 0.5 * w.squaredNorm() + regularisation_c * hinge_sum;

    if (grad) {
      // dL/dm = -2 * slack; gradient contribution per-sample is
      // -2 * slack_i * y_i * x_i for the `w` block, and
      // -2 * slack_i * y_i for the bias.
      const Eigen::VectorXd weighted_slacks =
          -2.0 * slacks.array() * labels.array();
      *grad = VectorType::Zero(d + 1);
      grad->head(d) =
          w + regularisation_c * (features.transpose() * weighted_slacks);
      (*grad)(d) = regularisation_c * weighted_slacks.sum();
    }
    return value;
  }
};

}  // namespace

int main() {
  const auto data = cppoptlib::examples::LoadIrisVersicolorVirginica();
  const Eigen::MatrixXd& features = data.features;
  const Eigen::VectorXd& labels = data.labels;
  const Eigen::Index feature_count = features.cols();
  const Eigen::Index variable_count = feature_count + 1;

  // Standard soft-margin strength; values in [0.1, 10] all give ~97%
  // accuracy on this split -- picking 1.0 for documentation.
  constexpr double regularisation_c = 1.0;

  SvmPrimalSquaredHinge objective(features, labels, regularisation_c);

  // Start at the origin: w = 0, b = 0.  Any feasible start works
  // because the objective is strongly convex for C < +inf.
  Eigen::VectorXd initial_x = Eigen::VectorXd::Zero(variable_count);

  cppoptlib::solver::Lbfgs<SvmPrimalSquaredHinge> solver;
  auto [solution, progress] =
      solver.Minimize(objective, cppoptlib::function::FunctionState(initial_x));

  const Eigen::VectorXd w = solution.x.head(feature_count);
  const double b = solution.x(feature_count);
  const Eigen::VectorXd scores = (features * w).array() + b;
  const double accuracy =
      cppoptlib::examples::ClassificationAccuracy(scores, labels);

  std::cout << "SVM primal (L-BFGS, squared hinge)\n";
  std::cout << "  solver status: " << progress.status << "\n";
  std::cout << "  iterations:    " << progress.num_iterations << "\n";
  std::cout << "  objective:     " << solution.value << "\n";
  std::cout << "  w:             " << w.transpose() << "\n";
  std::cout << "  b:             " << b << "\n";
  std::cout << "  accuracy:      " << accuracy << "\n";
  return 0;
}
