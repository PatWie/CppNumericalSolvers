// Copyright https://github.com/PatWie/CppNumericalSolvers, MIT license

#ifndef INCLUDE_CPPOPTLIB_FUNCTION_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_H_

#include <array>
#include <vector>

#include "Eigen/Core"
#include "utils/derivatives.h"

namespace cppoptlib {

namespace function {
template <class Scalar, class Vector>
struct State {
  Scalar value;
  Vector x;
  Vector gradient;

  template <class Function>
  void Update(const Function &function, const Vector &x0) {
    value = function(x0);
    x = x0;
    function.gradient(x0, &gradient);
  }
};

template <class TScalar, int TDim = Eigen::Dynamic>
class Function {
 public:
  static const int Dim = TDim;
  using Scalar = TScalar;
  using Vector = Eigen::Matrix<TScalar, Dim, 1>;
  using Hessian = Eigen::Matrix<TScalar, Dim, Dim>;
  using Index = typename Vector::Index;
  using Matrix = Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>;

 public:
  Function() = default;
  virtual ~Function() = default;

  virtual Scalar operator()(const Vector &x) const = 0;

  virtual void gradient(const Vector &x, Vector *grad) const {
    utils::ComputeFiniteGradient(*this, x, grad);
  }
  virtual void hessian(const Vector &x, Hessian *hessian) const {
    utils::ComputeFiniteHessian(*this, x, hessian);
  }
};
};  // namespace function

}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_FUNCTION_H_