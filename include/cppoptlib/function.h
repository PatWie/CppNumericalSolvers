// Copyright https://github.com/PatWie/CppNumericalSolvers, MIT license
#ifndef INCLUDE_CPPOPTLIB_FUNCTION_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_H_

#include "Eigen/Core"

namespace cppoptlib::function {

enum class Differentiability {
  None,   // Only function evaluation.
  First,  // Evaluation and gradient available.
  Second  // Evaluation, gradient, and Hessian available.
};

template <typename TFunc>
struct State {
  using function_t = TFunc;
};

template <class TScalar, int TDim>
class FunctionBase {
 public:
  using scalar_t = TScalar;
  using vector_t = Eigen::Matrix<TScalar, TDim, 1>;
  using matrix_t = Eigen::Matrix<TScalar, TDim, TDim>;
};

template <class TScalar, int TDim, Differentiability TDifferentiability>
class Function : public FunctionBase<TScalar, TDim> {
 public:
  using types = FunctionBase<TScalar, TDim>;
  using state_t = State<Function<TScalar, TDim, TDifferentiability>>;
};

template <class TScalar, int TDim>
struct State<Function<TScalar, TDim, Differentiability::None>> {
  using function_t = Function<TScalar, TDim, Differentiability::None>;
  using state_t = State<function_t>;

  typename function_t::scalar_t value = 0;
  typename function_t::vector_t x;

  State(const function_t &func, const typename function_t::vector_t &x_input)
      : x(x_input) {
    value = func(x_input);
  }

  State() {}

  State(const state_t &rhs) { CopyState(rhs); }  // nolint

  State operator=(const state_t &rhs) {
    CopyState(rhs);
    return *this;
  }

  void CopyState(const state_t &rhs) {
    value = rhs.value;
    x = rhs.x.eval();
  }
};

template <class TScalar, int TDim>
struct State<Function<TScalar, TDim, Differentiability::First>> {
  using function_t = Function<TScalar, TDim, Differentiability::First>;
  using state_t = State<function_t>;

  typename function_t::scalar_t value = 0;
  typename function_t::vector_t x;
  typename function_t::vector_t gradient;

  State(const function_t &func, const typename function_t::vector_t &x_input)
      : x(x_input), gradient(function_t::vector_t::Zero(x_input.rows())) {
    value = func(x_input);
    func.Gradient(x_input, &gradient);
  }

  State() {}

  State(const state_t &rhs) { CopyState(rhs); }  // nolint

  State operator=(const state_t &rhs) {
    CopyState(rhs);
    return *this;
  }

  void CopyState(const state_t &rhs) {
    value = rhs.value;
    x = rhs.x.eval();
    gradient = rhs.gradient.eval();
  }
};

template <class TScalar, int TDim>
struct State<Function<TScalar, TDim, Differentiability::Second>> {
  using function_t = Function<TScalar, TDim, Differentiability::Second>;
  using state_t = State<function_t>;

  typename function_t::scalar_t value = 0;
  typename function_t::vector_t x;
  typename function_t::vector_t gradient;
  typename function_t::matrix_t hessian;

  State(const function_t &func, const typename function_t::vector_t &x_input)
      : x(x_input),
        gradient(function_t::vector_t::Zero(x_input.rows())),
        hessian(function_t::matrix_t::Zero(x_input.rows(), x_input.rows())) {
    value = func(x_input);
    func.Gradient(x_input, &gradient);
    func.Hessian(x_input, &hessian);
  }

  State() {}

  State(const state_t &rhs) { CopyState(rhs); }  // nolint

  State operator=(const state_t &rhs) {
    CopyState(rhs);
    return *this;
  }

  void CopyState(const state_t &rhs) {
    value = rhs.value;
    x = rhs.x.eval();
    gradient = rhs.gradient.eval();
    hessian = rhs.hessian.eval();
  }
};

template <class TScalar, int TDim>
class Function<TScalar, TDim, Differentiability::None>
    : public FunctionBase<TScalar, TDim> {
 public:
  using types = FunctionBase<TScalar, TDim>;
  using state_t = State<Function<TScalar, TDim, Differentiability::None>>;
  static constexpr Differentiability diff_level = Differentiability::None;

  virtual typename types::scalar_t operator()(
      const typename types::vector_t &x) const = 0;
};

template <class TScalar, int TDim>
class Function<TScalar, TDim, Differentiability::First>
    : public Function<TScalar, TDim, Differentiability::None> {
 public:
  using types =
      typename Function<TScalar, TDim, Differentiability::None>::types;
  using state_t = State<Function<TScalar, TDim, Differentiability::First>>;
  static constexpr Differentiability diff_level = Differentiability::First;

  virtual void Gradient(const typename types::vector_t &x,
                        typename types::vector_t *grad) const = 0;
};

template <class TScalar, int TDim>
class Function<TScalar, TDim, Differentiability::Second>
    : public Function<TScalar, TDim, Differentiability::First> {
 public:
  using types =
      typename Function<TScalar, TDim, Differentiability::First>::types;
  using state_t = State<Function<TScalar, TDim, Differentiability::Second>>;
  static constexpr Differentiability diff_level = Differentiability::Second;

  virtual void Hessian(const typename types::vector_t &x,
                       typename types::matrix_t *hessian) const = 0;
};

}  // namespace cppoptlib::function

#endif  // INCLUDE_CPPOPTLIB_FUNCTION_H_
