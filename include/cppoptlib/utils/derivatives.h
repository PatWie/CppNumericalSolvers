// Copyright https://github.com/PatWie/CppNumericalSolvers, MIT license

#ifndef INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_
#define INCLUDE_CPPOPTLIB_UTILS_DERIVATIVES_H_

#include <algorithm>
#include <limits>
#include <vector>

namespace cppoptlib {
namespace utils {

// Approximates the gradient of the given function in x0.
template <class TFunction>
void ComputeFiniteGradient(const TFunction &function,
                           const typename TFunction::VectorT &x0,
                           typename TFunction::VectorT *grad,
                           const int accuracy = 0) {
  using ScalarT = typename TFunction::ScalarT;
  using VectorT = typename TFunction::VectorT;
  using IndexT = typename TFunction::IndexT;
  // The 'accuracy' can be 0, 1, 2, 3.
  constexpr ScalarT eps = 2.2204e-6;
  static const std::array<std::vector<ScalarT>, 4> coeff = {
      {{1, -1},
       {1, -8, 8, -1},
       {-1, 9, -45, 45, -9, 1},
       {3, -32, 168, -672, 672, -168, 32, -3}}};
  static const std::array<std::vector<ScalarT>, 4> coeff2 = {
      {{1, -1},
       {-2, -1, 1, 2},
       {-3, -2, -1, 1, 2, 3},
       {-4, -3, -2, -1, 1, 2, 3, 4}}};
  static const std::array<ScalarT, 4> dd = {2, 12, 60, 840};

  grad->resize(x0.rows());
  VectorT &x = const_cast<VectorT &>(x0);

  const int innerSteps = 2 * (accuracy + 1);
  const ScalarT ddVal = dd[accuracy] * eps;

  for (IndexT d = 0; d < x0.rows(); d++) {
    (*grad)[d] = 0;
    for (int s = 0; s < innerSteps; ++s) {
      ScalarT tmp = x[d];
      x[d] += coeff2[accuracy][s] * eps;
      (*grad)[d] += coeff[accuracy][s] * function(x);
      x[d] = tmp;
    }
    (*grad)[d] /= ddVal;
  }
}

// Approximates the HessianT of the given function in x0.
template <class TFunction>
void ComputeFiniteHessian(const TFunction &function,
                          const typename TFunction::VectorT &x0,
                          typename TFunction::HessianT *hessian,
                          int accuracy = 0) {
  using ScalarT = typename TFunction::ScalarT;
  using VectorT = typename TFunction::VectorT;
  using IndexT = typename TFunction::IndexT;

  constexpr ScalarT eps = std::numeric_limits<ScalarT>::epsilon() * 10e7;

  hessian->resize(x0.rows(), x0.rows());
  VectorT &x = const_cast<VectorT &>(x0);

  if (accuracy == 0) {
    for (IndexT i = 0; i < x0.rows(); i++) {
      for (IndexT j = 0; j < x0.rows(); j++) {
        ScalarT tmpi = x[i];
        ScalarT tmpj = x[j];

        ScalarT f4 = function(x);
        x[i] += eps;
        x[j] += eps;
        ScalarT f1 = function(x);
        x[j] -= eps;
        ScalarT f2 = function(x);
        x[j] += eps;
        x[i] -= eps;
        ScalarT f3 = function(x);
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
    for (IndexT i = 0; i < x0.rows(); i++) {
      for (IndexT j = 0; j < x0.rows(); j++) {
        ScalarT tmpi = x[i];
        ScalarT tmpj = x[j];

        ScalarT term_1 = 0;
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

        ScalarT term_2 = 0;
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

        ScalarT term_3 = 0;
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

        ScalarT term_4 = 0;
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

template <class TFunction>
bool IsGradientCorrect(const TFunction &function,
                       const typename TFunction::VectorT &x0,
                       int accuracy = 3) {
  constexpr float tolerance = 1e-2;

  using ScalarT = typename TFunction::ScalarT;
  using VectorT = typename TFunction::VectorT;
  using IndexT = typename TFunction::IndexT;

  const IndexT D = x0.rows();
  VectorT actual_gradient(D);
  VectorT expected_gradient(D);

  function.Gradient(x0, &actual_gradient);
  ComputeFiniteGradient(function, x0, &expected_gradient, accuracy);

  for (IndexT d = 0; d < D; ++d) {
    ScalarT scale =
        std::max(static_cast<ScalarT>(std::max(fabs(actual_gradient[d]),
                                               fabs(expected_gradient[d]))),
                 ScalarT(1.));
    if (fabs(actual_gradient[d] - expected_gradient[d]) > tolerance * scale)
      return false;
  }
  return true;
}

template <class TFunction>
bool IsHessianCorrect(const TFunction &function,
                      const typename TFunction::VectorT &x0, int accuracy = 3) {
  constexpr float tolerance = 1e-1;

  using ScalarT = typename TFunction::ScalarT;
  using HessianT = typename TFunction::HessianT;
  using IndexT = typename TFunction::IndexT;

  const IndexT D = x0.rows();

  HessianT actual_hessian = HessianT::Zero(D, D);
  HessianT expected_hessian = HessianT::Zero(D, D);
  function.Hessian(x0, &actual_hessian);
  ComputeFiniteHessian(function, x0, &expected_hessian, accuracy);
  for (IndexT d = 0; d < D; ++d) {
    for (IndexT e = 0; e < D; ++e) {
      ScalarT scale =
          std::max(static_cast<ScalarT>(std::max(fabs(actual_hessian(d, e)),
                                                 fabs(expected_hessian(d, e)))),
                   ScalarT(1.));
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
