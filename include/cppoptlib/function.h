// Copyright https://github.com/PatWie/CppNumericalSolvers, MIT license
#ifndef INCLUDE_CPPOPTLIB_FUNCTION_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_H_

#include "Eigen/Core"
#include "utils/derivatives.h"

namespace cppoptlib {
namespace function {

// Specifies a current function state.
template <class scalar_t, class vector_t, class matrix_t, int Ord>
struct State {
  scalar_t value = 0;                    // The objective value.
  vector_t x = vector_t::Zero();         // The current input value in x.
  vector_t gradient = vector_t::Zero();  // The gradient in x.
  matrix_t hessian = matrix_t::Zero();   // The Hessian in x;

  State() {}

  State operator=(const State<scalar_t, vector_t, matrix_t, 0> &rhs) {
    value = rhs.value;
    x = rhs.x.eval();
    return *this;
  }

  State operator=(const State<scalar_t, vector_t, matrix_t, 1> &rhs) {
    value = rhs.value;
    x = rhs.x.eval();
    gradient = rhs.gradient.eval();
    return *this;
  }

  State operator=(const State<scalar_t, vector_t, matrix_t, 2> &rhs) {
    value = rhs.value;
    x = rhs.x.eval();
    gradient = rhs.gradient.eval();
    hessian = rhs.hessian.eval();
    return *this;
  }
};

template <class TScalar, int Ord = 1, int TDim = Eigen::Dynamic>
class Function {
 public:
  static const int Dim = TDim;
  static const int Order = Ord;
  using scalar_t = TScalar;
  using vector_t = Eigen::Matrix<TScalar, Dim, 1>;
  using hessian_t = Eigen::Matrix<TScalar, Dim, Dim>;
  using index_t = typename vector_t::Index;
  using matrix_t = Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>;

  using state_t = function::State<scalar_t, vector_t, hessian_t, Order>;

 public:
  Function() = default;
  virtual ~Function() = default;

  // Computes the value of a function.
  virtual scalar_t operator()(const vector_t &x) const = 0;

  // Computes the gradient of a function.
  virtual void Gradient(const vector_t &x, vector_t *grad) const {
    utils::ComputeFiniteGradient(*this, x, grad);
  }
  // Computes the Hessian of a function.
  virtual void Hessian(const vector_t &x, hessian_t *hessian) const {
    utils::ComputeFiniteHessian(*this, x, hessian);
  }

  // For improved performance, this function will return the state directly.
  // Override this method if you can compute the objective value, gradient and
  // Hessian simultaneously.
  virtual State<scalar_t, vector_t, hessian_t, Ord> Eval(
      const vector_t &x) const {
    State<scalar_t, vector_t, hessian_t, Ord> state;
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

}  // namespace function
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_FUNCTION_H_
