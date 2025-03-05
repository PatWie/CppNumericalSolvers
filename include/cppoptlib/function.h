#ifndef INCLUDE_CPPOPTLIB_FUNCTION_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_H_

#include "Eigen/Core"

namespace cppoptlib::function {

enum class Differentiability {
  None,  // Only function evaluation.
  First, // Evaluation and gradient available.
  Second // Evaluation, gradient, and Hessian available.
};

template <typename TBase> struct State {
  using base_t = TBase;
  typename TBase::vector_t x;

  State() = default;
  State(const State &rhs) : x(rhs.x) {}
  State &operator=(const State &rhs) {
    if (this != &rhs) {
      x = rhs.x;
    }
    return *this;
  }
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

// Primary template declaration for CRTP-based Function.
template <class TScalar, int TDim, class TDerived,
          Differentiability TDifferentiability, int TNumConstraints = 0>
class Function;

// Specialization for functions with no derivative information.
template <class TScalar, int TDim, class TDerived, int TNumConstraints>
class Function<TScalar, TDim, TDerived, Differentiability::None,
               TNumConstraints>
    : public FunctionBase<TScalar, TDim, Differentiability::None,
                          TNumConstraints> {
public:
  using base_t =
      FunctionBase<TScalar, TDim, Differentiability::None, TNumConstraints>;
  using state_t = State<Function<TScalar, TDim, TDerived,
                                 Differentiability::None, TNumConstraints>>;

  using Derived = TDerived;

  inline typename base_t::scalar_t
  operator()(const typename base_t::vector_t &x) const {
    // Call the derived class method (which must be implemented as eval)
    return static_cast<const TDerived *>(this)->operator()(x);
  }

  state_t GetState(const typename base_t::vector_t &x) const {
    state_t state;
    state.x = x;
    return state;
  }
};

// Specialization for functions with gradient.
template <class TScalar, int TDim, class TDerived, int TNumConstraints>
class Function<TScalar, TDim, TDerived, Differentiability::First,
               TNumConstraints>
    : public FunctionBase<TScalar, TDim, Differentiability::First,
                          TNumConstraints> {
public:
  using base_t =
      FunctionBase<TScalar, TDim, Differentiability::First, TNumConstraints>;
  using state_t = State<Function<TScalar, TDim, TDerived,
                                 Differentiability::First, TNumConstraints>>;

  using Derived = TDerived;
  inline typename base_t::scalar_t
  operator()(const typename base_t::vector_t &x,
             typename base_t::vector_t *gradient = nullptr) const {
    return static_cast<const TDerived *>(this)->operator()(x, gradient);
  }

  state_t GetState(const typename base_t::vector_t &x) const {
    state_t state;
    state.x = x;
    return state;
  }
};

// Specialization for functions with Hessian.
template <class TScalar, int TDim, class TDerived, int TNumConstraints>
class Function<TScalar, TDim, TDerived, Differentiability::Second,
               TNumConstraints>
    : public FunctionBase<TScalar, TDim, Differentiability::Second,
                          TNumConstraints> {
public:
  using base_t =
      FunctionBase<TScalar, TDim, Differentiability::Second, TNumConstraints>;
  using state_t = State<Function<TScalar, TDim, TDerived,
                                 Differentiability::Second, TNumConstraints>>;

  using Derived = TDerived;
  inline typename base_t::scalar_t
  operator()(const typename base_t::vector_t &x,
             typename base_t::vector_t *gradient = nullptr,
             typename base_t::matrix_t *hessian = nullptr) const {
    return static_cast<const TDerived *>(this)->operator()(x, gradient,
                                                           hessian);
  }

  state_t GetState(const typename base_t::vector_t &x) const {
    state_t state;
    state.x = x;
    return state;
  }
};

template <typename F1, typename F2>
class AddFunction
    : public cppoptlib::function::Function<typename F1::scalar_t, F1::Dim,
                                           AddFunction<F1, F2>, F1::DiffLevel,
                                           F1::NumConstraints> {
public:
  using scalar_t = typename F1::scalar_t;
  using vector_t = typename F1::vector_t;
  using matrix_t =
      typename cppoptlib::function::Function<scalar_t, F1::Dim,
                                             AddFunction<F1, F2>, F1::DiffLevel,
                                             F1::NumConstraints>::matrix_t;

  // Static asserts to ensure f1 and f2 are compatible.
  static_assert(F1::DiffLevel == F2::DiffLevel,
                "Both functions must have the same differentiability level.");
  static_assert(F1::NumConstraints == F2::NumConstraints,
                "Both functions must have the same number of constraints.");
  static_assert(F1::Dim == F2::Dim,
                "Both functions must have the same input dimension.");

  AddFunction(const F1 &f1, const F2 &f2) : f1_(f1), f2_(f2) {}

  virtual scalar_t operator()(const vector_t &x, vector_t *gradient = nullptr,
                              matrix_t *hessian = nullptr) const {
    if constexpr (F1::DiffLevel ==
                  cppoptlib::function::Differentiability::None) {
      return f1_(x) + f2_(x);
    } else if constexpr (F1::DiffLevel ==
                         cppoptlib::function::Differentiability::First) {
      vector_t grad1, grad2;
      scalar_t val1 = f1_(x, gradient ? &grad1 : nullptr);
      scalar_t val2 = f2_(x, gradient ? &grad2 : nullptr);
      if (gradient) {
        *gradient = grad1 + grad2;
      }
      return val1 + val2;
    } else if constexpr (F1::DiffLevel ==
                         cppoptlib::function::Differentiability::Second) {
      vector_t grad1, grad2;
      matrix_t hess1, hess2;
      scalar_t val1 =
          f1_(x, gradient ? &grad1 : nullptr, hessian ? &hess1 : nullptr);
      scalar_t val2 =
          f2_(x, gradient ? &grad2 : nullptr, hessian ? &hess2 : nullptr);
      if (gradient) {
        *gradient = grad1 + grad2;
      }
      if (hessian) {
        *hessian = hess1 + hess2;
      }
      return val1 + val2;
    } else {
      // Should never get here.
      static_assert(always_false<F1>::value,
                    "Unsupported differentiability level in AddFunction.");
      return scalar_t(); // To silence warnings.
    }
  }

private:
  F1 f1_;
  F2 f2_;

  template <typename T> struct always_false : std::false_type {};
};

template <typename F1, typename F2>
AddFunction<F1, F2> operator+(const F1 &f1, const F2 &f2) {
  return AddFunction<F1, F2>(f1, f2);
}

template <typename F1, typename F2>
class SubFunction
    : public cppoptlib::function::Function<typename F1::scalar_t, F1::Dim,
                                           AddFunction<F1, F2>, F1::DiffLevel,
                                           F1::NumConstraints> {
public:
  using scalar_t = typename F1::scalar_t;
  using vector_t = typename F1::vector_t;
  using matrix_t =
      typename cppoptlib::function::Function<scalar_t, F1::Dim,
                                             AddFunction<F1, F2>, F1::DiffLevel,
                                             F1::NumConstraints>::matrix_t;

  // Static asserts to ensure f1 and f2 are compatible.
  static_assert(F1::DiffLevel == F2::DiffLevel,
                "Both functions must have the same differentiability level.");
  static_assert(F1::NumConstraints == F2::NumConstraints,
                "Both functions must have the same number of constraints.");
  static_assert(F1::Dim == F2::Dim,
                "Both functions must have the same input dimension.");

  SubFunction(const F1 &f1, const F2 &f2) : f1_(f1), f2_(f2) {}

  virtual scalar_t operator()(const vector_t &x, vector_t *gradient = nullptr,
                              matrix_t *hessian = nullptr) const {
    if constexpr (F1::DiffLevel ==
                  cppoptlib::function::Differentiability::None) {
      return f1_(x) - f2_(x);
    } else if constexpr (F1::DiffLevel ==
                         cppoptlib::function::Differentiability::First) {
      vector_t grad1, grad2;
      scalar_t val1 = f1_(x, gradient ? &grad1 : nullptr);
      scalar_t val2 = f2_(x, gradient ? &grad2 : nullptr);
      if (gradient) {
        *gradient = grad1 - grad2;
      }
      return val1 - val2;
    } else if constexpr (F1::DiffLevel ==
                         cppoptlib::function::Differentiability::Second) {
      vector_t grad1, grad2;
      matrix_t hess1, hess2;
      scalar_t val1 =
          f1_(x, gradient ? &grad1 : nullptr, hessian ? &hess1 : nullptr);
      scalar_t val2 =
          f2_(x, gradient ? &grad2 : nullptr, hessian ? &hess2 : nullptr);
      if (gradient) {
        *gradient = grad1 - grad2;
      }
      if (hessian) {
        *hessian = hess1 - hess2;
      }
      return val1 - val2;
    } else {
      // Should never get here.
      static_assert(always_false<F1>::value,
                    "Unsupported differentiability level in AddFunction.");
      return scalar_t(); // To silence warnings.
    }
  }

private:
  F1 f1_;
  F2 f2_;

  template <typename T> struct always_false : std::false_type {};
};

template <typename F1, typename F2>
SubFunction<F1, F2> operator-(const F1 &f1, const F2 &f2) {
  return SubFunction<F1, F2>(f1, f2);
}

template <typename F>
class MulFunction
    : public cppoptlib::function::Function<typename F::scalar_t, F::Dim,
                                           MulFunction<F>, F::DiffLevel,
                                           F::NumConstraints> {
public:
  using scalar_t = typename F::scalar_t;
  using vector_t = typename F::vector_t;
  using matrix_t =
      typename cppoptlib::function::Function<scalar_t, F::Dim, MulFunction<F>,
                                             F::DiffLevel,
                                             F::NumConstraints>::matrix_t;

  MulFunction(const F &f, scalar_t c) : f_(f), c_(c) {}

  virtual scalar_t operator()(const vector_t &x, vector_t *gradient = nullptr,
                              matrix_t *hessian = nullptr) const {
    if constexpr (F::DiffLevel ==
                  cppoptlib::function::Differentiability::None) {
      return c_ * f_(x);
    } else if constexpr (F::DiffLevel ==
                         cppoptlib::function::Differentiability::First) {
      vector_t grad;
      scalar_t val = f_(x, gradient ? &grad : nullptr);
      if (gradient) {
        *gradient = c_ * grad;
      }
      return c_ * val;
    } else if constexpr (F::DiffLevel ==
                         cppoptlib::function::Differentiability::Second) {
      vector_t grad;
      matrix_t hess;
      scalar_t val =
          f_(x, gradient ? &grad : nullptr, hessian ? &hess : nullptr);
      if (gradient) {
        *gradient = c_ * grad;
      }
      if (hessian) {
        *hessian = c_ * hess;
      }
      return c_ * val;
    } else {
      // Should never get here.
      static_assert(always_false<F>::value,
                    "Unsupported differentiability level in MulFunction.");
      return scalar_t(); // To silence warnings.
    }
  }

private:
  F f_;
  scalar_t c_;

  template <typename T> struct always_false : std::false_type {};
};

template <typename F>
MulFunction<F> operator*(const F &f, const typename F::scalar_t c) {
  return MulFunction<F>(f, c);
}

template <typename F>
MulFunction<F> operator*(const typename F::scalar_t c, const F &f) {
  return MulFunction<F>(f, c);
}

} // namespace cppoptlib::function

#endif // INCLUDE_CPPOPTLIB_FUNCTION_H_
