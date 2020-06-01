// Copyright https://github.com/PatWie/CppNumericalSolvers, MIT license

#ifndef INCLUDE_CPPOPTLIB_FUNCTION_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_H_

#include <array>
#include <vector>

#include "Eigen/Core"
#include "utils/derivatives.h"

namespace cppoptlib {

namespace function {

// Specifies a current function state.
template <class ScalarT, class VectorT, class MatrixT, int Ord>
struct State {
  ScalarT value = 0;                   // The objective value.
  VectorT x = VectorT::Zero();         // The current input value in x.
  VectorT gradient = VectorT::Zero();  // The gradient in x.
  MatrixT hessian = MatrixT::Zero();   // The Hessian in x;

  State() {}
  State(const State<ScalarT, VectorT, MatrixT, 0> &rhs) {
    value = rhs.value;
    x = rhs.x.eval();
  }

  State(const State<ScalarT, VectorT, MatrixT, 1> &rhs) {
    value = rhs.value;
    x = rhs.x.eval();
    gradient = rhs.gradient.eval();
  }
};

template <class TScalar, int Ord = 1, int TDim = Eigen::Dynamic>
class Function {
 public:
  static const int Dim = TDim;
  static const int Order = Ord;
  using ScalarT = TScalar;
  using VectorT = Eigen::Matrix<TScalar, Dim, 1>;
  using HessianT = Eigen::Matrix<TScalar, Dim, Dim>;
  using IndexT = typename VectorT::Index;
  using MatrixT = Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>;

 public:
  Function() = default;
  virtual ~Function() = default;

  // Computes the value of a function.
  virtual ScalarT operator()(const VectorT &x) const = 0;

  // Computes the gradient of a function.
  virtual void Gradient(const VectorT &x, VectorT *grad) const {
    utils::ComputeFiniteGradient(*this, x, grad);
  }
  // Computes the Hessian of a function.
  virtual void Hessian(const VectorT &x, HessianT *hessian) const {
    utils::ComputeFiniteHessian(*this, x, hessian);
  }

  // For improved performance, this function will return the state directly.
  State<ScalarT, VectorT, HessianT, Ord> Eval(const VectorT &x) const {
    State<ScalarT, VectorT, HessianT, Ord> state;
    state.value = this->operator()(x);
    state.x = x;
    if (Ord >= 1) {
      this->Gradient(x, &state.gradient);
    }
    if (Ord >= 2) {
      this->Hessian(x, &state.hessian);
    }
    return state;
  }
};
};  // namespace function

}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_FUNCTION_H_