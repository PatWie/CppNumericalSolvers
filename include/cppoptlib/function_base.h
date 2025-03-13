// Copyright https://github.com/PatWie/CppNumericalSolvers, MIT license
#ifndef INCLUDE_CPPOPTLIB_FUNCTION_BASE_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_BASE_H_

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace cppoptlib::function {

//-----------------------------------------------------------------
// Differentiability enum.
enum class DifferentiabilityMode {
  None = 0,  // Only function evaluation.
  First = 1, // Evaluation and gradient available.
  Second = 2 // Evaluation, gradient, and Hessian available.
};

//-----------------------------------------------------------------
// Unified Function Interface templated on scalar type and differentiability.
template <class TScalar, DifferentiabilityMode Mode,
          int TDimension = Eigen::Dynamic>
struct FunctionInterface {
  static constexpr int Dimension = TDimension;
  using ScalarType = TScalar;
  using VectorType = Eigen::Matrix<TScalar, Dimension, 1>;
  using MatrixType = Eigen::Matrix<TScalar, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability = Mode;

  // Unified signature: even if some derivatives are not computed.
  virtual ScalarType operator()(const VectorType &x, VectorType *grad = nullptr,
                                MatrixType *hess = nullptr) const = 0;
  virtual std::unique_ptr<FunctionInterface<TScalar, Mode, Dimension>>
  clone() const = 0;
  virtual ~FunctionInterface() = default;
};

//-----------------------------------------------------------------
// CRTP mixin that supplies the full interface from a simplified operator()().
// The derived class must implement one of the following simplified methods:
//   - For DifferentiabilityMode::None:   double operator()(const VectorType &x)
//   const;
//   - For DifferentiabilityMode::First:  double operator()(const VectorType &x,
//   VectorType *grad) const;
//   - For DifferentiabilityMode::Second: double operator()(const VectorType &x,
//   VectorType *grad, MatrixType *hess) const;
template <class Derived, class TScalar, DifferentiabilityMode TMode,
          int TDimension = Eigen::Dynamic>
struct FunctionCRTP : public FunctionInterface<TScalar, TMode, TDimension> {
  static constexpr int Dimension = TDimension;
  using ScalarType = TScalar;
  using VectorType = Eigen::Matrix<TScalar, Dimension, 1>;
  using MatrixType = Eigen::Matrix<TScalar, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability = TMode;

  virtual TScalar operator()(const VectorType &x, VectorType *grad = nullptr,
                             MatrixType *hess = nullptr) const override {
    if constexpr (TMode == DifferentiabilityMode::None) {
      return static_cast<const Derived *>(this)->operator()(x);
    } else if constexpr (TMode == DifferentiabilityMode::First) {
      if (hess) {
        throw std::runtime_error(
            "Hessian not available for first order function.");
      }
      return static_cast<const Derived *>(this)->operator()(x, grad);
    } else { // DifferentiabilityMode::Second
      return static_cast<const Derived *>(this)->operator()(x, grad, hess);
    }
  }

  virtual std::unique_ptr<FunctionInterface<TScalar, TMode, TDimension>>
  clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived &>(*this));
  }
};

//-----------------------------------------------------------------
// AnyFunction type erasure wrapper.

template <typename TScalar,
          DifferentiabilityMode TMode = DifferentiabilityMode::First,
          int TDimension = Eigen::Dynamic>
struct AnyFunction {
  static constexpr int Dimension = TDimension;
  using ScalarType = TScalar;
  using VectorType = Eigen::Matrix<TScalar, Dimension, 1>;
  using MatrixType = Eigen::Matrix<TScalar, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability = TMode;

  std::unique_ptr<FunctionInterface<TScalar, TMode, TDimension>> ptr;

  // Converting constructor: accepts any expression type by const reference,
  // provided it's not already an AnyFunction.
  template <typename F, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<F>, AnyFunction>>>
  AnyFunction(const F &f) : ptr(f.clone()) {
    static_assert(F::Differentiability == TMode,
                  "Differentiability mode mismatch");
    static_assert(F::Dimension == TDimension, "Dimension mismatch");
    static_assert(std::is_same<typename F::ScalarType, ScalarType>::value,
                  "Compile-time scalar-type mismatch");
  }

  // Copy constructor.
  AnyFunction(const AnyFunction &other)
      : ptr(other.ptr ? other.ptr->clone() : nullptr) {}

  // Copy assignment.
  AnyFunction &operator=(const AnyFunction &other) {
    if (this != &other)
      ptr = other.ptr ? other.ptr->clone() : nullptr;
    return *this;
  }

  // Default move constructor/assignment.
  AnyFunction(AnyFunction &&) noexcept = default;
  AnyFunction &operator=(AnyFunction &&) noexcept = default;

  // Call operator.
  ScalarType operator()(const VectorType &x, VectorType *grad = nullptr,
                        MatrixType *hess = nullptr) const {
    return (*ptr)(x, grad, hess);
  }
};

// Deduction guide for AnyFunction.

template <typename Expr>
AnyFunction(const Expr &)
    -> AnyFunction<typename Expr::ScalarType, Expr::Differentiability,
                   Expr::Dimension>;

// template <typename Expr>
// AnyFunction(const Expr &expr) {
//   // Use SFINAE to ensure that Expr meets the necessary interface.
//   static_assert(std::is_convertible<decltype(expr(std::declval<typename
//   AnyFunction::VectorType>())),
//                                     typename AnyFunction::ScalarType>::value,
//                 "Expression does not meet required interface");
//   ptr = expr.clone(); // Assuming your expressions have a clone() method.
// }

template <typename TScalar, int TDimension = Eigen::Dynamic>
struct FunctionState {
  static constexpr int NumConstraints = 0;
  using VectorType = Eigen::Matrix<TScalar, TDimension, 1>;
  VectorType x;

  explicit FunctionState(VectorType x) : x(x) {}
};

template <typename TScalar, int TDimension>
FunctionState(Eigen::Matrix<TScalar, TDimension, 1>)
    -> FunctionState<TScalar, TDimension>;
} // namespace cppoptlib::function

#endif //  INCLUDE_CPPOPTLIB_FUNCTION_BASE_H_
