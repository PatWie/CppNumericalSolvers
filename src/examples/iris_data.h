// Iris versicolor-vs-virginica binary classification dataset.
//
// Fisher's Iris dataset, 50 versicolor samples (label -1) followed by
// 50 virginica samples (label +1).  Features are sepal length, sepal
// width, petal length, petal width, in centimetres.  Setosa is
// excluded because it is linearly separable from the other two in
// the original 4-D feature space -- too easy to distinguish the
// hinge-loss and dual SVM formulations.  Versicolor vs. virginica is
// the canonical non-trivially-separable split used throughout the
// SVM literature.
//
// Data is embedded as `constexpr std::array` so the example binaries
// do not do any file I/O; the dataset is tiny (100 rows x 4 floats =
// 3.2 kB).  Values are the standard UCI Iris numbers.
#ifndef SRC_EXAMPLES_IRIS_DATA_H_
#define SRC_EXAMPLES_IRIS_DATA_H_

#include <Eigen/Core>
#include <array>
#include <cstddef>

namespace cppoptlib::examples {

// Number of samples in the versicolor+virginica subset.
constexpr std::size_t iris_sample_count = 100;
// Feature dimensionality (sepal length, sepal width, petal length, petal
// width).
constexpr std::size_t iris_feature_count = 4;

// 100 samples x 4 features, row-major.  Rows 0..49 are versicolor;
// rows 50..99 are virginica.
constexpr std::array<std::array<double, iris_feature_count>, iris_sample_count>
    iris_raw_features = {{
        // ---- Versicolor (label = -1) ----
        {{7.0, 3.2, 4.7, 1.4}},
        {{6.4, 3.2, 4.5, 1.5}},
        {{6.9, 3.1, 4.9, 1.5}},
        {{5.5, 2.3, 4.0, 1.3}},
        {{6.5, 2.8, 4.6, 1.5}},
        {{5.7, 2.8, 4.5, 1.3}},
        {{6.3, 3.3, 4.7, 1.6}},
        {{4.9, 2.4, 3.3, 1.0}},
        {{6.6, 2.9, 4.6, 1.3}},
        {{5.2, 2.7, 3.9, 1.4}},
        {{5.0, 2.0, 3.5, 1.0}},
        {{5.9, 3.0, 4.2, 1.5}},
        {{6.0, 2.2, 4.0, 1.0}},
        {{6.1, 2.9, 4.7, 1.4}},
        {{5.6, 2.9, 3.6, 1.3}},
        {{6.7, 3.1, 4.4, 1.4}},
        {{5.6, 3.0, 4.5, 1.5}},
        {{5.8, 2.7, 4.1, 1.0}},
        {{6.2, 2.2, 4.5, 1.5}},
        {{5.6, 2.5, 3.9, 1.1}},
        {{5.9, 3.2, 4.8, 1.8}},
        {{6.1, 2.8, 4.0, 1.3}},
        {{6.3, 2.5, 4.9, 1.5}},
        {{6.1, 2.8, 4.7, 1.2}},
        {{6.4, 2.9, 4.3, 1.3}},
        {{6.6, 3.0, 4.4, 1.4}},
        {{6.8, 2.8, 4.8, 1.4}},
        {{6.7, 3.0, 5.0, 1.7}},
        {{6.0, 2.9, 4.5, 1.5}},
        {{5.7, 2.6, 3.5, 1.0}},
        {{5.5, 2.4, 3.8, 1.1}},
        {{5.5, 2.4, 3.7, 1.0}},
        {{5.8, 2.7, 3.9, 1.2}},
        {{6.0, 2.7, 5.1, 1.6}},
        {{5.4, 3.0, 4.5, 1.5}},
        {{6.0, 3.4, 4.5, 1.6}},
        {{6.7, 3.1, 4.7, 1.5}},
        {{6.3, 2.3, 4.4, 1.3}},
        {{5.6, 3.0, 4.1, 1.3}},
        {{5.5, 2.5, 4.0, 1.3}},
        {{5.5, 2.6, 4.4, 1.2}},
        {{6.1, 3.0, 4.6, 1.4}},
        {{5.8, 2.6, 4.0, 1.2}},
        {{5.0, 2.3, 3.3, 1.0}},
        {{5.6, 2.7, 4.2, 1.3}},
        {{5.7, 3.0, 4.2, 1.2}},
        {{5.7, 2.9, 4.2, 1.3}},
        {{6.2, 2.9, 4.3, 1.3}},
        {{5.1, 2.5, 3.0, 1.1}},
        {{5.7, 2.8, 4.1, 1.3}},
        // ---- Virginica (label = +1) ----
        {{6.3, 3.3, 6.0, 2.5}},
        {{5.8, 2.7, 5.1, 1.9}},
        {{7.1, 3.0, 5.9, 2.1}},
        {{6.3, 2.9, 5.6, 1.8}},
        {{6.5, 3.0, 5.8, 2.2}},
        {{7.6, 3.0, 6.6, 2.1}},
        {{4.9, 2.5, 4.5, 1.7}},
        {{7.3, 2.9, 6.3, 1.8}},
        {{6.7, 2.5, 5.8, 1.8}},
        {{7.2, 3.6, 6.1, 2.5}},
        {{6.5, 3.2, 5.1, 2.0}},
        {{6.4, 2.7, 5.3, 1.9}},
        {{6.8, 3.0, 5.5, 2.1}},
        {{5.7, 2.5, 5.0, 2.0}},
        {{5.8, 2.8, 5.1, 2.4}},
        {{6.4, 3.2, 5.3, 2.3}},
        {{6.5, 3.0, 5.5, 1.8}},
        {{7.7, 3.8, 6.7, 2.2}},
        {{7.7, 2.6, 6.9, 2.3}},
        {{6.0, 2.2, 5.0, 1.5}},
        {{6.9, 3.2, 5.7, 2.3}},
        {{5.6, 2.8, 4.9, 2.0}},
        {{7.7, 2.8, 6.7, 2.0}},
        {{6.3, 2.7, 4.9, 1.8}},
        {{6.7, 3.3, 5.7, 2.1}},
        {{7.2, 3.2, 6.0, 1.8}},
        {{6.2, 2.8, 4.8, 1.8}},
        {{6.1, 3.0, 4.9, 1.8}},
        {{6.4, 2.8, 5.6, 2.1}},
        {{7.2, 3.0, 5.8, 1.6}},
        {{7.4, 2.8, 6.1, 1.9}},
        {{7.9, 3.8, 6.4, 2.0}},
        {{6.4, 2.8, 5.6, 2.2}},
        {{6.3, 2.8, 5.1, 1.5}},
        {{6.1, 2.6, 5.6, 1.4}},
        {{7.7, 3.0, 6.1, 2.3}},
        {{6.3, 3.4, 5.6, 2.4}},
        {{6.4, 3.1, 5.5, 1.8}},
        {{6.0, 3.0, 4.8, 1.8}},
        {{6.9, 3.1, 5.4, 2.1}},
        {{6.7, 3.1, 5.6, 2.4}},
        {{6.9, 3.1, 5.1, 2.3}},
        {{5.8, 2.7, 5.1, 1.9}},
        {{6.8, 3.2, 5.9, 2.3}},
        {{6.7, 3.3, 5.7, 2.5}},
        {{6.7, 3.0, 5.2, 2.3}},
        {{6.3, 2.5, 5.0, 1.9}},
        {{6.5, 3.0, 5.2, 2.0}},
        {{6.2, 3.4, 5.4, 2.3}},
        {{5.9, 3.0, 5.1, 1.8}},
    }};

// Labels matching `iris_raw_features` row-for-row.  Encoded as int so
// the value -1 is unambiguous; the loader below casts to the scalar
// type the examples want.
constexpr std::array<int, iris_sample_count> iris_raw_labels = {{
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1,
    +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1,
    +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1,
    +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1,
}};

// Dataset bundle: feature matrix (n x d) and label vector (n).
// Labels are strict {-1, +1} doubles so the SVM margin `y * (w^T x + b)`
// is a signed scalar product.
struct IrisDataset {
  Eigen::MatrixXd features;  // 100 x 4, optionally z-score normalised.
  Eigen::Matrix<double, Eigen::Dynamic, 1> labels;  // 100 x 1, values +/- 1.
};

// Load the versicolor-vs-virginica subset.  When `normalize == true`
// (the default), each feature column is centred and scaled to unit
// standard deviation.  Normalisation keeps the margin constraint
// `y_i (w^T x_i + b) >= 1` comparable across formulations -- without
// it the raw petal-length/width values dominate and `C` needs
// re-tuning per formulation.
inline IrisDataset LoadIrisVersicolorVirginica(bool normalize = true) {
  IrisDataset data;
  data.features.resize(static_cast<Eigen::Index>(iris_sample_count),
                       static_cast<Eigen::Index>(iris_feature_count));
  data.labels.resize(static_cast<Eigen::Index>(iris_sample_count));
  for (std::size_t i = 0; i < iris_sample_count; ++i) {
    for (std::size_t j = 0; j < iris_feature_count; ++j) {
      data.features(static_cast<Eigen::Index>(i),
                    static_cast<Eigen::Index>(j)) = iris_raw_features[i][j];
    }
    data.labels(static_cast<Eigen::Index>(i)) =
        static_cast<double>(iris_raw_labels[i]);
  }
  if (normalize) {
    const Eigen::RowVectorXd mean = data.features.colwise().mean();
    data.features.rowwise() -= mean;
    // Sample standard deviation (divide by n - 1), population is fine
    // for this example but sample matches sklearn's default.
    const double scale_normaliser =
        1.0 / std::sqrt(static_cast<double>(iris_sample_count - 1));
    const Eigen::RowVectorXd standard_deviation =
        (data.features.array().square().colwise().sum().sqrt() *
         scale_normaliser)
            .matrix();
    for (Eigen::Index j = 0; j < data.features.cols(); ++j) {
      if (standard_deviation(j) > 1e-12) {
        data.features.col(j) /= standard_deviation(j);
      }
    }
  }
  return data;
}

// Classifier-accuracy helper shared across the SVM examples.  `scores`
// is the signed margin `w^T x + b`; a positive value predicts class +1.
inline double ClassificationAccuracy(
    const Eigen::VectorXd& scores,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& labels) {
  Eigen::Index correct_count = 0;
  for (Eigen::Index i = 0; i < scores.size(); ++i) {
    const double predicted_sign = scores(i) >= 0.0 ? 1.0 : -1.0;
    if (predicted_sign == labels(i)) {
      ++correct_count;
    }
  }
  return static_cast<double>(correct_count) /
         static_cast<double>(scores.size());
}

}  // namespace cppoptlib::examples

#endif  // SRC_EXAMPLES_IRIS_DATA_H_
