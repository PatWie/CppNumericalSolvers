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

/**
 * @brief Forward declaration for State.
 *
 * The State structure is used to encapsulate the result of a function
 * evaluation, including the function value, the point at which it was
 * evaluated, and, if available, its derivatives (gradient and Hessian).
 *
 * @tparam TBase The base type whose state is being stored.
 */
template <typename TBase>
struct State {
  using base_t = TBase;
};

template <class TScalar, int TDim, Differentiability TDifferentiability,
          int TNumConstraints = 0>
class FunctionBase {
 public:
  using scalar_t = TScalar;
  using vector_t = Eigen::Matrix<TScalar, TDim, 1>;
  using matrix_t = Eigen::Matrix<TScalar, TDim, TDim>;
  static constexpr Differentiability DiffLevel = TDifferentiability;
  static constexpr int NumConstraints = TNumConstraints;
  static constexpr int Dim = TDim;
};

template <class TScalar, int TDim, Differentiability TDifferentiability>
class Function : public FunctionBase<TScalar, TDim, TDifferentiability> {
 public:
  using base_t = FunctionBase<TScalar, TDim, TDifferentiability>;
  using state_t = State<Function<TScalar, TDim, TDifferentiability>>;
};

template <class TScalar, int TDim>
struct State<FunctionBase<TScalar, TDim, Differentiability::None>> {
  using base_t = FunctionBase<TScalar, TDim, Differentiability::None>;
  using state_t = State<base_t>;

  typename base_t::scalar_t value = 0;
  typename base_t::vector_t x;

  State() = default;

  State(const state_t &rhs) : value(rhs.value), x(rhs.x.eval()) {}

  state_t &operator=(const state_t &rhs) {
    if (this != &rhs) {
      value = rhs.value;
      x = rhs.x.eval();
    }
    return *this;
  }
};

template <class TScalar, int TDim>
struct State<FunctionBase<TScalar, TDim, Differentiability::First>> {
  using base_t = FunctionBase<TScalar, TDim, Differentiability::First>;
  using state_t = State<base_t>;

  typename base_t::scalar_t value = 0;
  typename base_t::vector_t x;
  typename base_t::vector_t gradient;

  State() = default;

  State(const state_t &rhs)
      : value(rhs.value), x(rhs.x.eval()), gradient(rhs.gradient.eval()) {}

  state_t &operator=(const state_t &rhs) {
    if (this != &rhs) {
      value = rhs.value;
      x = rhs.x.eval();
      gradient = rhs.gradient.eval();
    }
    return *this;
  }
};

template <class TScalar, int TDim>
struct State<FunctionBase<TScalar, TDim, Differentiability::Second>> {
  using base_t = FunctionBase<TScalar, TDim, Differentiability::Second>;
  using state_t = State<base_t>;

  typename base_t::scalar_t value = 0;
  typename base_t::vector_t x;
  typename base_t::vector_t gradient;
  typename base_t::matrix_t hessian;

  State() = default;

  State(const state_t &rhs)
      : value(rhs.value),
        x(rhs.x.eval()),
        gradient(rhs.gradient.eval()),
        hessian(rhs.hessian.eval()) {}

  state_t &operator=(const state_t &rhs) {
    if (this != &rhs) {
      value = rhs.value;
      x = rhs.x.eval();
      gradient = rhs.gradient.eval();
      hessian = rhs.hessian.eval();
    }
    return *this;
  }
};

/**
 * @brief Function interface for functions with no derivative information.
 *
 * This class must be derived from by user-defined functions that only provide a
 * function evaluation (no gradient or Hessian).
 *
 * @tparam TScalar The scalar type.
 * @tparam TDim The dimension of the input vector.
 */
template <class TScalar, int TDim>
class Function<TScalar, TDim, Differentiability::None>
    : public FunctionBase<TScalar, TDim, Differentiability::None> {
 public:
  using base_t = FunctionBase<TScalar, TDim, Differentiability::None>;
  using state_t = State<base_t>;

  virtual typename base_t::scalar_t operator()(
      const typename base_t::vector_t &x) const = 0;

  state_t GetState(const typename base_t::vector_t &x) const {
    state_t state;
    state.x = x;
    state.value = this->operator()(x);
    return state;
  }
};

/**
 * @brief Function interface for functions with gradient information.
 *
 * User-defined functions that provide both function evaluation and gradient
 * computation should derive from this class.
 *
 * @tparam TScalar The scalar type.
 * @tparam TDim The dimension of the input vector.
 */
template <class TScalar, int TDim>
class Function<TScalar, TDim, Differentiability::First>
    : public FunctionBase<TScalar, TDim, Differentiability::First> {
 public:
  using base_t = FunctionBase<TScalar, TDim, Differentiability::First>;
  using state_t = State<base_t>;

  virtual typename base_t::scalar_t operator()(
      const typename base_t::vector_t &x,
      typename base_t::vector_t *gradient = nullptr) const = 0;
  state_t GetState(const typename base_t::vector_t &x) const {
    state_t state;
    state.x = x;
    state.value = this->operator()(x, &state.gradient);
    return state;
  }
};

/**
 * @brief Function interface for functions with Hessian information.
 *
 * User-defined functions that provide function value, gradient, and Hessian
 * computations should derive from this class.
 *
 * @tparam TScalar The scalar type.
 * @tparam TDim The dimension of the input vector.
 */
template <class TScalar, int TDim>
class Function<TScalar, TDim, Differentiability::Second>
    : public FunctionBase<TScalar, TDim, Differentiability::Second> {
 public:
  using base_t = FunctionBase<TScalar, TDim, Differentiability::Second>;
  using state_t = State<base_t>;

  virtual typename base_t::scalar_t operator()(
      const typename base_t::vector_t &x,
      typename base_t::vector_t *gradient = nullptr,
      typename base_t::matrix_t *hessian = nullptr) const = 0;
  state_t GetState(const typename base_t::vector_t &x) const {
    state_t state;
    state.x = x;
    state.value = this->operator()(x, &state.gradient, &state.hessian);
    return state;
  }
};

}  // namespace cppoptlib::function

#endif  // INCLUDE_CPPOPTLIB_FUNCTION_H_
