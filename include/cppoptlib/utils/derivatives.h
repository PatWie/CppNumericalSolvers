// Copyright https://github.com/PatWie/CppNumericalSolvers, MIT license

#ifndef INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_
#define INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_

#include <algorithm>
#include <limits>
#include <vector>
#include <array>

namespace cppoptlib {
namespace utils {

// Approximates the gradient of the given function in x0.
template <class function_t>
void ComputeFiniteGradient(const function_t &function,
                           const typename function_t::vector_t &x0,
                           typename function_t::vector_t *grad,
                           const int accuracy = 0) {
  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  using index_t = typename function_t::index_t;
  // The 'accuracy' can be 0, 1, 2, 3.
  constexpr scalar_t eps = 2.2204e-6;
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
  vector_t &x = const_cast<vector_t &>(x0);

  const int innerSteps = 2 * (accuracy + 1);
  const scalar_t ddVal = dd[accuracy] * eps;

  for (index_t d = 0; d < x0.rows(); d++) {
    (*grad)[d] = 0;
    for (int s = 0; s < innerSteps; ++s) {
      scalar_t tmp = x[d];
      x[d] += coeff2[accuracy][s] * eps;
      (*grad)[d] += coeff[accuracy][s] * function(x);
      x[d] = tmp;
    }
    (*grad)[d] /= ddVal;
  }
}

// Approximates the hessian_t of the given function in x0.
template <class function_t>
void ComputeFiniteHessian(const function_t &function,
                          const typename function_t::vector_t &x0,
                          typename function_t::hessian_t *hessian,
                          int accuracy = 0) {
  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  using index_t = typename function_t::index_t;

  constexpr scalar_t eps = std::numeric_limits<scalar_t>::epsilon() * 10e7;

  hessian->resize(x0.rows(), x0.rows());
  vector_t &x = const_cast<vector_t &>(x0);

  if (accuracy == 0) {
    for (index_t i = 0; i < x0.rows(); i++) {
      for (index_t j = 0; j < x0.rows(); j++) {
        scalar_t tmpi = x[i];
        scalar_t tmpj = x[j];

        scalar_t f4 = function(x);
        x[i] += eps;
        x[j] += eps;
        scalar_t f1 = function(x);
        x[j] -= eps;
        scalar_t f2 = function(x);
        x[j] += eps;
        x[i] -= eps;
        scalar_t f3 = function(x);
        (*hessian)(i, j) = (f1 - f2 - f3 + f4) / (eps * eps);

        x[i] = tmpi;
        x[j] = tmpj;
      }
    }
  } else {
    /*
      \displaystyle{{\frac{\partial^2{f}}{\partial{x0}\partial{y}}}\approx
      \frac{1}{600\,h^2} \left[\begin{matrix}
        -63(f_{1,-2}+f_{2,-1}+f_{-2,1}+f_{-1,2})+\\
          63(f_{-1,-2}+f_{-2,-1}+f_{1,2}+f_{2,1})+\\
          44(f_{2,-2}+f_{-2,2}-f_{-2,-2}-f_{2,2})+\\
          74(f_{-1,-1}+f_{1,1}-f_{1,-1}-f_{-1,1})
      \end{matrix}\right] }
    */
    for (index_t i = 0; i < x0.rows(); i++) {
      for (index_t j = 0; j < x0.rows(); j++) {
        scalar_t tmpi = x[i];
        scalar_t tmpj = x[j];

        scalar_t term_1 = 0;
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += 1 * eps;
        x[j] += -2 * eps;
        term_1 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += 2 * eps;
        x[j] += -1 * eps;
        term_1 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += -2 * eps;
        x[j] += 1 * eps;
        term_1 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += -1 * eps;
        x[j] += 2 * eps;
        term_1 += function(x);

        scalar_t term_2 = 0;
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += -1 * eps;
        x[j] += -2 * eps;
        term_2 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += -2 * eps;
        x[j] += -1 * eps;
        term_2 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += 1 * eps;
        x[j] += 2 * eps;
        term_2 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += 2 * eps;
        x[j] += 1 * eps;
        term_2 += function(x);

        scalar_t term_3 = 0;
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += 2 * eps;
        x[j] += -2 * eps;
        term_3 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += -2 * eps;
        x[j] += 2 * eps;
        term_3 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += -2 * eps;
        x[j] += -2 * eps;
        term_3 -= function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += 2 * eps;
        x[j] += 2 * eps;
        term_3 -= function(x);

        scalar_t term_4 = 0;
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += -1 * eps;
        x[j] += -1 * eps;
        term_4 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += 1 * eps;
        x[j] += 1 * eps;
        term_4 += function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += 1 * eps;
        x[j] += -1 * eps;
        term_4 -= function(x);
        x[i] = tmpi;
        x[j] = tmpj;
        x[i] += -1 * eps;
        x[j] += 1 * eps;
        term_4 -= function(x);

        x[i] = tmpi;
        x[j] = tmpj;

        (*hessian)(i, j) =
            (-63 * term_1 + 63 * term_2 + 44 * term_3 + 74 * term_4) /
            (600.0 * eps * eps);
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
  using index_t = typename function_t::index_t;

  const index_t D = x0.rows();
  vector_t actual_gradient(D);
  vector_t expected_gradient(D);

  function.Gradient(x0, &actual_gradient);
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
                      const typename function_t::vector_t &x0, int accuracy = 3) {
  constexpr float tolerance = 1e-1;

  using scalar_t = typename function_t::scalar_t;
  using hessian_t = typename function_t::hessian_t;
  using index_t = typename function_t::index_t;

  const index_t D = x0.rows();

  hessian_t actual_hessian = hessian_t::Zero(D, D);
  hessian_t expected_hessian = hessian_t::Zero(D, D);
  function.Hessian(x0, &actual_hessian);
  ComputeFiniteHessian(function, x0, &expected_hessian, accuracy);
  for (index_t d = 0; d < D; ++d) {
    for (index_t e = 0; e < D; ++e) {
      scalar_t scale =
          std::max(static_cast<scalar_t>(std::max(fabs(actual_hessian(d, e)),
                                                 fabs(expected_hessian(d, e)))),
                   scalar_t(1.));
      if (fabs(actual_hessian(d, e) - expected_hessian(d, e)) >
          tolerance * scale)
        return false;
    }
  }
  return true;
}

};  // namespace utils
};  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_
