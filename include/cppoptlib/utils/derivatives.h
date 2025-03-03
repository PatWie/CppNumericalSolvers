// Copyright https://github.com/PatWie/CppNumericalSolvers, MIT license
#ifndef INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_
#define INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace cppoptlib::utils {

template <class function_t>
void ComputeFiniteGradient(
    const function_t &function,
    const Eigen::Matrix<typename function_t::scalar_t, Eigen::Dynamic, 1> &x0,
    Eigen::Matrix<typename function_t::scalar_t, Eigen::Dynamic, 1> *grad,
    const int accuracy = 0) {
  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  using index_t = typename vector_t::Index;

  constexpr scalar_t machine_eps = std::numeric_limits<scalar_t>::epsilon();

  // Coefficients for finite difference formulas of increasing accuracy.
  // The 'accuracy' can be 0, 1, 2, or 3.
  static const std::array<std::vector<scalar_t>, 4> coeff = {
      {{1, -1},
       {1, -8, 8, -1},
       {-1, 9, -45, 45, -9, 1},
       {3, -32, 168, -672, 672, -168, 32, -3}}};
  static const std::array<std::vector<scalar_t>, 4> coeff2 = {
      {{1, -1},
       {-2, -1, 1, 2},
       {-3, -2, -1, 1, 2, 3},
       {-4, -3, -2, -1, 1, 2, 3, 4}}};

  static const std::array<scalar_t, 4> dd = {2, 12, 60, 840};

  grad->resize(x0.rows());
  vector_t x = x0;

  const int innerSteps = 2 * (accuracy + 1);
  for (index_t d = 0; d < x0.rows(); d++) {
    // Compute a coordinate-dependent step size.
    scalar_t h =
        std::sqrt(machine_eps) * std::max(std::abs(x0[d]), scalar_t(1));
    scalar_t ddVal = dd[accuracy] * h;
    (*grad)[d] = 0;
    for (int s = 0; s < innerSteps; ++s) {
      scalar_t tmp = x[d];
      x[d] += coeff2[accuracy][s] * h;
      (*grad)[d] += coeff[accuracy][s] * function(x);
      x[d] = tmp;
    }
    (*grad)[d] /= ddVal;
  }
}

// Approximates the Hessian of the given function in x0.
template <class function_t>
void ComputeFiniteHessian(
    const function_t &function,
    const Eigen::Matrix<typename function_t::scalar_t, Eigen::Dynamic, 1> &x0,
    Eigen::Matrix<typename function_t::scalar_t, Eigen::Dynamic, Eigen::Dynamic>
        *hessian,
    int accuracy = 0) {
  using scalar_t = typename function_t::scalar_t;
  using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
  using index_t = typename vector_t::Index;

  constexpr scalar_t machine_eps = std::numeric_limits<scalar_t>::epsilon();
  index_t n = x0.rows();
  hessian->resize(n, n);

  // Make a local copy so we can safely modify it.
  vector_t x = x0;
  scalar_t f0 = function(x0);

  if (accuracy == 0) {
    // Basic central difference approximation.
    for (index_t i = 0; i < n; i++) {
      // Choose an adaptive step size for the i-th coordinate.
      scalar_t hi =
          std::sqrt(machine_eps) * std::max(std::abs(x0[i]), scalar_t(1));
      // Diagonal: standard second derivative.
      x = x0;
      x[i] += hi;
      scalar_t f_plus = function(x);
      x = x0;
      x[i] -= hi;
      scalar_t f_minus = function(x);
      (*hessian)(i, i) = (f_plus - 2 * f0 + f_minus) / (hi * hi);

      for (index_t j = i + 1; j < n; j++) {
        scalar_t hj =
            std::sqrt(machine_eps) * std::max(std::abs(x0[j]), scalar_t(1));
        // Off-diagonals: use central differences.
        x = x0;
        x[i] += hi;
        x[j] += hj;
        scalar_t f_pp = function(x);
        x = x0;
        x[i] += hi;
        x[j] -= hj;
        scalar_t f_pm = function(x);
        x = x0;
        x[i] -= hi;
        x[j] += hj;
        scalar_t f_mp = function(x);
        x = x0;
        x[i] -= hi;
        x[j] -= hj;
        scalar_t f_mm = function(x);
        scalar_t d2f = (f_pp - f_pm - f_mp + f_mm) / (4 * hi * hj);
        (*hessian)(i, j) = d2f;
        (*hessian)(j, i) = d2f;
      }
    }
  } else {
    // Higher-order finite difference approximation using
    for (index_t i = 0; i < n; i++) {
      scalar_t hi =
          std::sqrt(machine_eps) * std::max(std::abs(x0[i]), scalar_t(1));
      x = x0;
      x[i] += hi;
      scalar_t f_plus = function(x);
      x = x0;
      x[i] -= hi;
      scalar_t f_minus = function(x);
      (*hessian)(i, i) = (f_plus - 2 * f0 + f_minus) / (hi * hi);

      for (index_t j = i + 1; j < n; j++) {
        scalar_t hj =
            std::sqrt(machine_eps) * std::max(std::abs(x0[j]), scalar_t(1));
        scalar_t h = (hi + hj) / 2;

        scalar_t tmpi = x0[i], tmpj = x0[j];
        scalar_t term1 = 0, term2 = 0, term3 = 0, term4 = 0;

        // term1: -63 * (f(tmpi+1h, tmpj-2h) + f(tmpi+2h, tmpj-1h) + f(tmpi-2h,
        // tmpj+1h) + f(tmpi-1h, tmpj+2h))
        x = x0;
        x[i] = tmpi + 1 * h;
        x[j] = tmpj - 2 * h;
        term1 += function(x);
        x = x0;
        x[i] = tmpi + 2 * h;
        x[j] = tmpj - 1 * h;
        term1 += function(x);
        x = x0;
        x[i] = tmpi - 2 * h;
        x[j] = tmpj + 1 * h;
        term1 += function(x);
        x = x0;
        x[i] = tmpi - 1 * h;
        x[j] = tmpj + 2 * h;
        term1 += function(x);

        // term2: +63 * (f(tmpi-1h, tmpj-2h) + f(tmpi-2h, tmpj-1h) + f(tmpi+1h,
        // tmpj+2h) + f(tmpi+2h, tmpj+1h))
        x = x0;
        x[i] = tmpi - 1 * h;
        x[j] = tmpj - 2 * h;
        term2 += function(x);
        x = x0;
        x[i] = tmpi - 2 * h;
        x[j] = tmpj - 1 * h;
        term2 += function(x);
        x = x0;
        x[i] = tmpi + 1 * h;
        x[j] = tmpj + 2 * h;
        term2 += function(x);
        x = x0;
        x[i] = tmpi + 2 * h;
        x[j] = tmpj + 1 * h;
        term2 += function(x);

        // term3: +44 * (f(tmpi+2h, tmpj-2h) + f(tmpi-2h, tmpj+2h) - f(tmpi-2h,
        // tmpj-2h) - f(tmpi+2h, tmpj+2h))
        x = x0;
        x[i] = tmpi + 2 * h;
        x[j] = tmpj - 2 * h;
        term3 += function(x);
        x = x0;
        x[i] = tmpi - 2 * h;
        x[j] = tmpj + 2 * h;
        term3 += function(x);
        x = x0;
        x[i] = tmpi - 2 * h;
        x[j] = tmpj - 2 * h;
        term3 -= function(x);
        x = x0;
        x[i] = tmpi + 2 * h;
        x[j] = tmpj + 2 * h;
        term3 -= function(x);

        // term4: +74 * (f(tmpi-1h, tmpj-1h) + f(tmpi+1h, tmpj+1h) - f(tmpi+1h,
        // tmpj-1h) - f(tmpi-1h, tmpj+1h))
        x = x0;
        x[i] = tmpi - 1 * h;
        x[j] = tmpj - 1 * h;
        term4 += function(x);
        x = x0;
        x[i] = tmpi + 1 * h;
        x[j] = tmpj + 1 * h;
        term4 += function(x);
        x = x0;
        x[i] = tmpi + 1 * h;
        x[j] = tmpj - 1 * h;
        term4 -= function(x);
        x = x0;
        x[i] = tmpi - 1 * h;
        x[j] = tmpj + 1 * h;
        term4 -= function(x);

        // Combine the weighted terms.
        scalar_t mixed = (-63 * term1 + 63 * term2 + 44 * term3 + 74 * term4) /
                         (600 * h * h);
        (*hessian)(i, j) = mixed;
        (*hessian)(j, i) = mixed;
      }
    }
  }
}

template <class function_t>
bool IsGradientCorrect(const function_t &function,
                       const typename function_t::vector_t &x0,
                       int accuracy = 3) {
  constexpr float tolerance = 1e-2;

  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  using index_t = typename vector_t::Index;

  const index_t D = x0.rows();
  vector_t actual_gradient;
  function(x0, &actual_gradient);
  vector_t expected_gradient(D);

  ComputeFiniteGradient(function, x0, &expected_gradient, accuracy);

  for (index_t d = 0; d < D; ++d) {
    scalar_t scale =
        std::max(static_cast<scalar_t>(std::max(fabs(actual_gradient[d]),
                                                fabs(expected_gradient[d]))),
                 scalar_t(1.));
    if (fabs(actual_gradient[d] - expected_gradient[d]) > tolerance * scale)
      return false;
  }
  return true;
}

template <class function_t>
bool IsHessianCorrect(const function_t &function,
                      const typename function_t::vector_t &x0,
                      int accuracy = 3) {
  constexpr float tolerance = 1e-1;

  using scalar_t = typename function_t::scalar_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;
  using index_t = typename vector_t::Index;

  const index_t D = x0.rows();

  matrix_t actual_hessian;
  function(x0, nullptr, &actual_hessian);
  matrix_t expected_hessian = matrix_t::Zero(D, D);
  ComputeFiniteHessian(function, x0, &expected_hessian, accuracy);
  for (index_t d = 0; d < D; ++d) {
    for (index_t e = 0; e < D; ++e) {
      scalar_t scale = std::max(
          static_cast<scalar_t>(std::max(fabs(actual_hessian(d, e)),
                                         fabs(expected_hessian(d, e)))),
          scalar_t(1.));
      if (fabs(actual_hessian(d, e) - expected_hessian(d, e)) >
          tolerance * scale)
        return false;
    }
  }
  return true;
}

}  // namespace cppoptlib::utils

#endif  // INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_
