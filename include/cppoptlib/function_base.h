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
// FunctionCRTP inherits `operator()(x, grad, hess)` from
// `FunctionInterface`, but derived user classes typically supply a
// shorter-arity `operator()(x, grad)` (or `operator()(x)`).  Under the
// usual C++ hiding rules that shorter overload hides the inherited
// three-arg virtual, which gcc-15 reports as `-Woverloaded-virtual` at
// every single derived instantiation.  The warning is structural to
// this CRTP design: the library's public shape requires the derived
// class to define `operator()`, yet doing so triggers the warning.
//
// The textbook cure -- non-virtual interface, users override a
// differently-named `DoEvaluate` -- would break the public API of every
// existing user class.  Suppress the warning here and only here; every
// derived class inherits the pragma scope of this template definition.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#endif

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

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

//-----------------------------------------------------------------
// FunctionExpr type erasure wrapper.

// Internal: adapter that presents a higher-mode `FunctionInterface` as a
// lower-mode one.  Used by `FunctionExpr`'s converting constructor to
// accept a `Second`-mode expression into a `First`-mode or `None`-mode
// wrapper.  The adapter stores the original interface by
// `unique_ptr<FunctionInterface<T, SourceMode, Dim>>` and forwards
// `operator()` calls, discarding the fields the target mode does not
// expose.  Downgrading from `Second` to `First` is safe: the Hessian
// pointer is simply never passed through; the inner caller was going
// to ignore it anyway (no first-order solver reads Hessians from the
// composite).  Downgrading from `Second` to `None` similarly drops the
// gradient and Hessian pointers.  Downgrading from `First` to `None`
// drops only the gradient.  Upgrades (lower to higher) are refused at
// compile time in `FunctionExpr`'s converting constructor; there is no
// way to synthesise a gradient or Hessian out of thin air.
template <typename TScalar, DifferentiabilityMode SourceMode,
          DifferentiabilityMode TargetMode, int TDimension>
struct ModeDowngradeAdapter
    : public FunctionInterface<TScalar, TargetMode, TDimension> {
  static_assert(static_cast<int>(SourceMode) >= static_cast<int>(TargetMode),
                "ModeDowngradeAdapter only lowers the differentiability mode "
                "-- attempting to upgrade.");

  using VectorType = Eigen::Matrix<TScalar, TDimension, 1>;
  using MatrixType = Eigen::Matrix<TScalar, TDimension, TDimension>;

  std::unique_ptr<FunctionInterface<TScalar, SourceMode, TDimension>> source;

  explicit ModeDowngradeAdapter(
      std::unique_ptr<FunctionInterface<TScalar, SourceMode, TDimension>> s)
      : source(std::move(s)) {}

  TScalar operator()(const VectorType& x, VectorType* grad = nullptr,
                     MatrixType* hess = nullptr) const override {
    (void)hess;  // Never forwarded -- the target mode does not expose it.
    if constexpr (TargetMode == DifferentiabilityMode::None) {
      (void)grad;
      return (*source)(x, nullptr, nullptr);
    } else if constexpr (TargetMode == DifferentiabilityMode::First) {
      return (*source)(x, grad, nullptr);
    } else {
      // TargetMode == Second: SourceMode must also be Second by the
      // static_assert above, in which case the adapter is a no-op and
      // nothing should ever construct it at this branch.  Keep the path
      // defined in case some future code wires it up.
      return (*source)(x, grad, hess);
    }
  }

  std::unique_ptr<FunctionInterface<TScalar, TargetMode, TDimension>> clone()
      const override {
    return std::make_unique<
        ModeDowngradeAdapter<TScalar, SourceMode, TargetMode, TDimension>>(
        source->clone());
  }
};

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

  // Converting constructor.  Accepts any non-FunctionExpr expression
  // whose differentiability is at least as strong as `TMode` -- that is,
  // we let a `Second`-mode source land in a `First`-mode wrapper (or a
  // `First`-mode source land in a `None`-mode wrapper), discarding the
  // extra derivative information.  The static_asserts refuse upgrades.
  template <typename F, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<F>, FunctionExpr>>>
  FunctionExpr(const F& f) {
    static_assert(
        static_cast<int>(F::Differentiability) >= static_cast<int>(TMode),
        "Differentiability mode mismatch: source must supply at "
        "least as much derivative information as the target mode "
        "requires (downgrades are accepted, upgrades are not).");
    static_assert(F::Dimension == TDimension, "Dimension mismatch");
    static_assert(std::is_same<typename F::ScalarType, ScalarType>::value,
                  "Compile-time scalar-type mismatch");
    if constexpr (F::Differentiability == TMode) {
      ptr = f.clone();
    } else {
      // Downgrade adapter: wrap the stronger-mode clone so the stored
      // pointer type matches `FunctionInterface<TScalar, TMode, Dim>`.
      auto source =
          f.clone();  // unique_ptr<FunctionInterface<T, F::Mode, Dim>>
      ptr = std::make_unique<ModeDowngradeAdapter<TScalar, F::Differentiability,
                                                  TMode, TDimension>>(
          std::move(source));
    }
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

  // Exposes the stored interface for mode-downgrading converting
  // construction into a weaker-mode `FunctionExpr`.  Returns a
  // deep-copied interface pointer so the caller owns independent
  // lifetime; `FunctionExpr` itself does not implement
  // `FunctionInterface` so this is not `virtual`.
  std::unique_ptr<FunctionInterface<TScalar, TMode, TDimension>> clone() const {
    return ptr ? ptr->clone() : nullptr;
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
  // Plain unconstrained state; `Progress::Update` uses this to pick the
  // gradient/f-delta convergence branch rather than the feasibility
  // branch that `AugmentedLagrangeState` takes.
  static constexpr bool IsConstrained = false;
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
