// CPPNumericalSolvers - A lightweight C++ numerical optimization library
// Copyright (c) 2014    Patrick Wieschollek + Contributors
// Licensed under the MIT License (see below).
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Author: Patrick Wieschollek
//
// More details can be found in the project documentation:
// https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_FUNCTION_BASE_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_BASE_H_

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cppoptlib::function {

//-----------------------------------------------------------------
// Differentiability enum.
enum class DifferentiabilityMode {
  None = 0,   // Only function evaluation.
  First = 1,  // Evaluation and gradient available.
  Second = 2  // Evaluation, gradient, and Hessian available.
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
  virtual ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                                MatrixType* hess = nullptr) const = 0;
  virtual std::unique_ptr<FunctionInterface<TScalar, Mode, Dimension>> clone()
      const = 0;
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

  virtual TScalar operator()(const VectorType& x, VectorType* grad = nullptr,
                             MatrixType* hess = nullptr) const override {
    if constexpr (TMode == DifferentiabilityMode::None) {
      return static_cast<const Derived*>(this)->operator()(x);
    } else if constexpr (TMode == DifferentiabilityMode::First) {
      if (hess) {
        throw std::runtime_error(
            "Hessian not available for first order function.");
      }
      return static_cast<const Derived*>(this)->operator()(x, grad);
    } else {  // DifferentiabilityMode::Second
      return static_cast<const Derived*>(this)->operator()(x, grad, hess);
    }
  }

  virtual std::unique_ptr<FunctionInterface<TScalar, TMode, TDimension>> clone()
      const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

//-----------------------------------------------------------------
// FunctionExpr type erasure wrapper.

template <typename TScalar,
          DifferentiabilityMode TMode = DifferentiabilityMode::First,
          int TDimension = Eigen::Dynamic>
struct FunctionExpr {
  static constexpr int Dimension = TDimension;
  using ScalarType = TScalar;
  using VectorType = Eigen::Matrix<TScalar, Dimension, 1>;
  using MatrixType = Eigen::Matrix<TScalar, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability = TMode;

  std::unique_ptr<FunctionInterface<TScalar, TMode, TDimension>> ptr;

  // Converting constructor: accepts any expression type by const reference,
  // provided it's not already an FunctionExpr.
  template <typename F, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<F>, FunctionExpr>>>
  FunctionExpr(const F& f) : ptr(f.clone()) {
    static_assert(F::Differentiability == TMode,
                  "Differentiability mode mismatch");
    static_assert(F::Dimension == TDimension, "Dimension mismatch");
    static_assert(std::is_same<typename F::ScalarType, ScalarType>::value,
                  "Compile-time scalar-type mismatch");
  }

  // Copy constructor.
  FunctionExpr(const FunctionExpr& other)
      : ptr(other.ptr ? other.ptr->clone() : nullptr) {}

  // Copy assignment.
  FunctionExpr& operator=(const FunctionExpr& other) {
    if (this != &other) ptr = other.ptr ? other.ptr->clone() : nullptr;
    return *this;
  }

  // Default move constructor/assignment.
  FunctionExpr(FunctionExpr&&) noexcept = default;
  FunctionExpr& operator=(FunctionExpr&&) noexcept = default;

  // Call operator.
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                        MatrixType* hess = nullptr) const {
    return (*ptr)(x, grad, hess);
  }
};

// Deduction guide for FunctionExpr.

template <typename Expr>
FunctionExpr(const Expr&)
    -> FunctionExpr<typename Expr::ScalarType, Expr::Differentiability,
                    Expr::Dimension>;

// template <typename Expr>
// FunctionExpr(const Expr &expr) {
//   // Use SFINAE to ensure that Expr meets the necessary interface.
//   static_assert(std::is_convertible<decltype(expr(std::declval<typename
//   FunctionExpr::VectorType>())),
//                                     typename
//                                     FunctionExpr::ScalarType>::value,
//                 "Expression does not meet required interface");
//   ptr = expr.clone(); // Assuming your expressions have a clone() method.
// }

// A point along the optimization trajectory, together with the objective
// value and gradient at that point.  The invariant callers rely on is:
// **every FunctionState carries the value and gradient at its `x`.**  Only
// two places construct a FunctionState that leaves this invariant intact:
//
//   1. The `(function, x)` constructor below, which evaluates once and
//      stores all fields.
//   2. The `(x, value, gradient)` "trust-me" constructor, used by the line
//      search to hand back the accepted step without re-evaluating: the
//      line search already computed those numbers on its last internal
//      trial point.
//
// Components that consume a FunctionState (Progress::Update, callbacks,
// the outer solver loop) therefore read `state.value` and `state.gradient`
// directly instead of calling `function(state.x, ...)` again.  This
// eliminates the "three redundant function calls per iteration" pattern
// that showed up in traces against libLBFGS and L-BFGS-B Fortran 3.0.
template <typename TScalar, int TDimension = Eigen::Dynamic>
struct FunctionState {
  static constexpr int NumConstraints = 0;
  using ScalarType = TScalar;
  using VectorType = Eigen::Matrix<TScalar, TDimension, 1>;

  VectorType x;
  ScalarType value = ScalarType{0};
  VectorType gradient;  // empty for DifferentiabilityMode::None.

  // Legacy x-only constructor.  Leaves value = 0 and gradient = empty; the
  // caller must populate them before the state is consumed by the new
  // eval-free code paths.  Kept so existing user code that writes
  // `FunctionState(x0)` continues to compile; the outer solver `Minimize`
  // will evaluate `function(x)` once and rebuild a fully-populated state.
  explicit FunctionState(VectorType x_in) : x(std::move(x_in)) {}

  // Evaluate `function` at `x` once and populate all fields.
  template <class FunctionT>
  FunctionState(const FunctionT& function, VectorType x_in)
      : x(std::move(x_in)) {
    if constexpr (FunctionT::Differentiability == DifferentiabilityMode::None) {
      value = function(x);
    } else {
      value = function(x, &gradient);
    }
  }

  // Trust-me constructor: caller has already computed `value` and `gradient`
  // at `x` (typically the line search's last internal evaluation).
  FunctionState(VectorType x_in, ScalarType value_in, VectorType gradient_in)
      : x(std::move(x_in)), value(value_in), gradient(std::move(gradient_in)) {}
};

template <typename TScalar, int TDimension>
FunctionState(Eigen::Matrix<TScalar, TDimension, 1>)
    -> FunctionState<TScalar, TDimension>;
}  // namespace cppoptlib::function

#endif  //  INCLUDE_CPPOPTLIB_FUNCTION_BASE_H_
