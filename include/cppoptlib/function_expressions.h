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
#ifndef INCLUDE_CPPOPTLIB_FUNCTION_EXPRESSIONS_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_EXPRESSIONS_H_

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace cppoptlib::function {
//-----------------------------------------------------------------
// Expression Templates for Binary Operations.
// These also derive from the unified interface.

// ConstExpression: a constant function.
// We use DifferentiabilityMode::Second so that grad and hess are always
// provided.
template <typename TScalar,
          DifferentiabilityMode TMode = DifferentiabilityMode::Second,
          int TDimension = Eigen::Dynamic>
struct ConstExpression : public FunctionInterface<TScalar, TMode, TDimension> {
  static constexpr int Dimension = TDimension;
  using ScalarType = TScalar;
  using VectorType = Eigen::Matrix<TScalar, Dimension, 1>;
  using MatrixType = Eigen::Matrix<TScalar, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability = TMode;

  TScalar c;
  explicit ConstExpression(TScalar c_) : c(c_) {}
  virtual TScalar operator()(const VectorType& x, VectorType* grad = nullptr,
                             MatrixType* hess = nullptr) const override {
    (void)x;
    if (grad) {
      *grad = VectorType::Zero(x.size());
    }
    if (hess) {
      *hess = MatrixType::Zero(x.size(), x.size());
    }
    return c;
  }
  virtual std::unique_ptr<FunctionInterface<TScalar, TMode, Dimension>> clone()
      const override {
    return std::make_unique<ConstExpression<TScalar, TMode, Dimension>>(c);
  }
};

// Helper to compute the minimum differentiability mode.
template <typename F, typename G>
struct MinDifferentiability {
  static constexpr DifferentiabilityMode value =
      (static_cast<int>(F::Differentiability) <
               static_cast<int>(G::Differentiability)
           ? F::Differentiability
           : G::Differentiability);
};
// Meta-function to compute the minimum differentiability mode.
template <DifferentiabilityMode A, DifferentiabilityMode B>
struct MinDifferentiabilityMode {
  static constexpr DifferentiabilityMode value =
      (static_cast<int>(A) < static_cast<int>(B) ? A : B);
};

// Addition Expression.
template <typename F, typename G,
          DifferentiabilityMode TMode = MinDifferentiability<F, G>::value>
struct AddExpression
    : public FunctionInterface<typename F::ScalarType, TMode, F::Dimension> {
  static_assert(
      F::Dimension == G::Dimension,
      "Compile-time dimension mismatch: F and G must have the same dimension.");
  static_assert(
      std::is_same<typename F::ScalarType, typename G::ScalarType>::value,
      "Compile-time scalar-type mismatch: F and G must have the same scalar "
      "type.");

  static constexpr int Dimension = F::Dimension;
  using ScalarType = typename F::ScalarType;
  using VectorType = Eigen::Matrix<ScalarType, Dimension, 1>;
  using MatrixType = Eigen::Matrix<ScalarType, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability = TMode;

  F f;
  G g;
  AddExpression(const F& f_, const G& g_) : f(f_), g(g_) {}
  virtual ScalarType operator()(
      const VectorType& x, [[maybe_unused]] VectorType* grad = nullptr,
      [[maybe_unused]] MatrixType* hess = nullptr) const override {
    if constexpr (TMode == DifferentiabilityMode::None) {
      return f(x) + g(x);
    } else if constexpr (TMode == DifferentiabilityMode::First) {
      VectorType grad_f(x.size()), grad_g(x.size());
      ScalarType fx_f = f(x, &grad_f);
      ScalarType fx_g = g(x, &grad_g);
      if (grad) {
        *grad = grad_f + grad_g;
      }
      return fx_f + fx_g;
    } else {  // Second
      VectorType grad_f(x.size()), grad_g(x.size());
      MatrixType hess_f(x.size(), x.size()), hess_g(x.size(), x.size());
      ScalarType fx_f = f(x, &grad_f, &hess_f);
      ScalarType fx_g = g(x, &grad_g, &hess_g);
      if (grad) {
        *grad = grad_f + grad_g;
      }
      if (hess) {
        *hess = hess_f + hess_g;
      }
      return fx_f + fx_g;
    }
  }
  virtual std::unique_ptr<FunctionInterface<ScalarType, TMode, Dimension>>
  clone() const override {
    return std::make_unique<AddExpression<F, G, TMode>>(f, g);
  }
};

// Subtraction Expression.
template <typename F, typename G,
          DifferentiabilityMode TMode = MinDifferentiability<F, G>::value>
struct SubExpression
    : public FunctionInterface<typename F::ScalarType, TMode, F::Dimension> {
  static_assert(
      F::Dimension == G::Dimension,
      "Compile-time dimension mismatch: F and G must have the same dimension.");
  static_assert(
      std::is_same<typename F::ScalarType, typename G::ScalarType>::value,
      "Compile-time scalar-type mismatch: F and G must have the same scalar "
      "type.");

  static constexpr int Dimension = F::Dimension;
  using ScalarType = typename F::ScalarType;
  using VectorType = Eigen::Matrix<ScalarType, Dimension, 1>;
  using MatrixType = Eigen::Matrix<ScalarType, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability = TMode;

  F f;
  G g;
  SubExpression(const F& f_, const G& g_) : f(f_), g(g_) {}
  virtual ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                                MatrixType* hess = nullptr) const override {
    if constexpr (TMode == DifferentiabilityMode::None) {
      return f(x) - g(x);
    } else if constexpr (TMode == DifferentiabilityMode::First) {
      VectorType grad_f(x.size()), grad_g(x.size());
      ScalarType fx_f = f(x, &grad_f);
      ScalarType fx_g = g(x, &grad_g);
      if (grad) {
        *grad = grad_f - grad_g;
      }
      return fx_f - fx_g;
    } else {  // Second
      VectorType grad_f(x.size()), grad_g(x.size());
      MatrixType hess_f(x.size(), x.size()), hess_g(x.size(), x.size());
      ScalarType fx_f = f(x, &grad_f, &hess_f);
      ScalarType fx_g = g(x, &grad_g, &hess_g);
      if (grad) {
        *grad = grad_f - grad_g;
      }
      if (hess) {
        *hess = hess_f - hess_g;
      }
      return fx_f - fx_g;
    }
  }
  virtual std::unique_ptr<FunctionInterface<ScalarType, TMode, Dimension>>
  clone() const override {
    return std::make_unique<SubExpression<F, G, TMode>>(f, g);
  }
};

// Multiplication by a Constant Expression.
template <typename F, typename TScalar = typename F::ScalarType,
          DifferentiabilityMode TMode = F::Differentiability>
struct MulExpression : public FunctionInterface<TScalar, TMode, F::Dimension> {
  static_assert(
      std::is_same<typename F::ScalarType, TScalar>::value,
      "Compile-time scalar-type mismatch: F and c must have the same scalar "
      "type.");

  static constexpr int Dimension = F::Dimension;
  using ScalarType = typename F::ScalarType;
  using VectorType = Eigen::Matrix<ScalarType, Dimension, 1>;
  using MatrixType = Eigen::Matrix<ScalarType, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability = TMode;

  TScalar c;
  F f;
  MulExpression(const TScalar& c_, const F& f_) : c(c_), f(f_) {}
  virtual ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                                MatrixType* hess = nullptr) const override {
    if (c == TScalar(0)) {
      if (grad) {
        *grad = VectorType::Zero(x.size());
      }
      if (hess) {
        *hess = MatrixType::Zero(x.size(), x.size());
      }
      return TScalar(0);
    }
    if constexpr (TMode == DifferentiabilityMode::None) {
      return c * f(x);
    } else if constexpr (TMode == DifferentiabilityMode::First) {
      VectorType grad_f(x.size());
      ScalarType fx = f(x, &grad_f);
      if (grad) {
        *grad = c * grad_f;
      }
      return c * fx;
    } else {  // Second
      VectorType grad_f(x.size());
      MatrixType hess_f(x.size(), x.size());
      ScalarType fx = f(x, &grad_f, &hess_f);
      if (grad) {
        *grad = c * grad_f;
      }
      if (hess) {
        *hess = c * hess_f;
      }
      return c * fx;
    }
  }
  virtual std::unique_ptr<FunctionInterface<TScalar, TMode, Dimension>> clone()
      const override {
    return std::make_unique<MulExpression<F, TScalar, TMode>>(c, f);
  }
};

//-------------------------------------------------------------
// Product Expression: represents the product of two functions.
template <typename F, typename G,
          DifferentiabilityMode TMode = MinDifferentiability<F, G>::value>
struct ProdExpression
    : public FunctionInterface<typename F::ScalarType, TMode, F::Dimension> {
  static_assert(
      F::Dimension == G::Dimension,
      "Compile-time dimension mismatch: F and G must have the same dimension.");
  static_assert(
      std::is_same<typename F::ScalarType, typename G::ScalarType>::value,
      "Compile-time scalar-type mismatch: F and G must have the same scalar "
      "type.");

  static constexpr int Dimension = F::Dimension;
  using ScalarType = typename F::ScalarType;
  using VectorType = Eigen::Matrix<ScalarType, Dimension, 1>;
  using MatrixType = Eigen::Matrix<ScalarType, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability = TMode;

  F f;
  G g;

  ProdExpression(const F& f_, const G& g_) : f(f_), g(g_) {}

  virtual ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                                MatrixType* hess = nullptr) const override {
    if constexpr (TMode == DifferentiabilityMode::None) {
      return f(x) * g(x);
    } else if constexpr (TMode == DifferentiabilityMode::First) {
      VectorType grad_f(x.size()), grad_g(x.size());
      ScalarType fx = f(x, &grad_f);
      ScalarType gx = g(x, &grad_g);
      if (grad) {
        // Product rule: (f*g)' = f'*g + f*g'
        *grad = gx * grad_f + fx * grad_g;
      }
      return fx * gx;
    } else {  // Second order
      VectorType grad_f(x.size()), grad_g(x.size());
      MatrixType hess_f(x.size(), x.size()), hess_g(x.size(), x.size());
      ScalarType fx = f(x, &grad_f, &hess_f);
      ScalarType gx = g(x, &grad_g, &hess_g);
      if (grad) {
        *grad = gx * grad_f + fx * grad_g;
      }
      if (hess) {
        // Hessian: f''*g + f'*g' (symmetric) + f*g''
        *hess = gx * hess_f + fx * hess_g + grad_f * grad_g.transpose() +
                grad_g * grad_f.transpose();
      }
      return fx * gx;
    }
  }

  virtual std::unique_ptr<FunctionInterface<ScalarType, TMode, Dimension>>
  clone() const override {
    return std::make_unique<ProdExpression<F, G, TMode>>(f, g);
  }
};

// --- Helper Expression for min{0, f(x)}
template <typename F>
struct MinZeroExpression
    : public FunctionInterface<typename F::ScalarType, F::Differentiability,
                               F::Dimension> {
  static constexpr int Dimension = F::Dimension;
  using ScalarType = typename F::ScalarType;
  using VectorType = Eigen::Matrix<ScalarType, Dimension, 1>;
  using MatrixType = Eigen::Matrix<ScalarType, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability =
      F::Differentiability;

  F f;
  explicit MinZeroExpression(const F& f_) : f(f_) {}

  virtual ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                                MatrixType* hess = nullptr) const override {
    if constexpr (Differentiability == DifferentiabilityMode::None) {
      ScalarType val = f(x);
      // Use ConstExpression to return zero (with zero derivatives) if inactive.
      return (val >= 0)
                 ? ConstExpression<ScalarType, Differentiability, Dimension>(
                       ScalarType(0))(x)
                 : val;
    } else {
      ScalarType val = f(x, grad, hess);
      if (val >= 0) {
        // Return a constant zero value along with zero gradient/Hessian.
        return ConstExpression<ScalarType, Differentiability, Dimension>(
            ScalarType(0))(x, grad, hess);
      } else {
        return val;
      }
    }
  }

  virtual std::unique_ptr<
      FunctionInterface<ScalarType, Differentiability, Dimension>>
  clone() const override {
    return std::make_unique<MinZeroExpression<F>>(f);
  }
};

// --- Helper Expression for max{0, f(x)}
template <typename F>
struct MaxZeroExpression
    : public FunctionInterface<typename F::ScalarType, F::Differentiability,
                               F::Dimension> {
  static constexpr int Dimension = F::Dimension;
  using ScalarType = typename F::ScalarType;
  using VectorType = Eigen::Matrix<ScalarType, Dimension, 1>;
  using MatrixType = Eigen::Matrix<ScalarType, Dimension, Dimension>;
  static constexpr DifferentiabilityMode Differentiability =
      F::Differentiability;

  F f;
  explicit MaxZeroExpression(const F& f_) : f(f_) {}

  virtual ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                                MatrixType* hess = nullptr) const override {
    if constexpr (Differentiability == DifferentiabilityMode::None) {
      ScalarType val = f(x);
      return (val <= 0)
                 ? ConstExpression<ScalarType, Differentiability, Dimension>(
                       ScalarType(0))(x)
                 : val;
    } else {
      ScalarType val = f(x, grad, hess);
      if (val <= 0) {
        return ConstExpression<ScalarType, Differentiability, Dimension>(
            ScalarType(0))(x, grad, hess);
      } else {
        return val;
      }
    }
  }

  virtual std::unique_ptr<
      FunctionInterface<ScalarType, Differentiability, Dimension>>
  clone() const override {
    return std::make_unique<MaxZeroExpression<F>>(f);
  }
};

// For addition.
//
template <typename F, typename G,
          typename = std::void_t<decltype(F::Differentiability),
                                 decltype(G::Differentiability)>>
auto operator+(const F& f, const G& g) {
  static_assert(
      std::is_same<typename F::ScalarType, typename G::ScalarType>::value,
      "ScalarType must match in addition.");
  static_assert(
      F::Dimension == G::Dimension,
      "Dimension mismatch: F and G must have the same compile-time dimension.");
  constexpr auto TMode = MinDifferentiability<F, G>::value;
  return AddExpression<F, G, TMode>(f, g);
}

// For subtraction.
template <typename F, typename G,
          typename = std::void_t<decltype(F::Differentiability),
                                 decltype(G::Differentiability)>>
auto operator-(const F& f, const G& g) {
  static_assert(
      std::is_same<typename F::ScalarType, typename G::ScalarType>::value,
      "ScalarType must match in addition.");
  static_assert(
      F::Dimension == G::Dimension,
      "Dimension mismatch: F and G must have the same compile-time dimension.");
  constexpr auto TMode = MinDifferentiability<F, G>::value;
  return SubExpression<F, G, TMode>(f, g);
}

// For multiplication (F * c and c * F).
template <typename F>
auto operator*(const F& f, const typename F::ScalarType& c)
    -> MulExpression<F, typename F::ScalarType, F::Differentiability> {
  using ScalarType = typename F::ScalarType;
  // Always return a MulExpression. Inside its operator() you can check if c==0.
  return MulExpression<F, ScalarType, F::Differentiability>(c, f);
}

template <typename F>
auto operator*(const typename F::ScalarType& c, const F& f)
    -> MulExpression<F, typename F::ScalarType, F::Differentiability> {
  using ScalarType = typename F::ScalarType;
  return MulExpression<F, ScalarType, F::Differentiability>(c, f);
}

//-------------------------------------------------------------
// Free operator overload for the product of two functions.
template <typename F, typename G,
          typename = std::void_t<decltype(F::Differentiability),
                                 decltype(G::Differentiability)>>
auto operator*(const F& f, const G& g) {
  static_assert(
      std::is_same<typename F::ScalarType, typename G::ScalarType>::value,
      "ScalarType must match in addition.");
  static_assert(
      F::Dimension == G::Dimension,
      "Dimension mismatch: F and G must have the same compile-time dimension.");
  constexpr auto TMode = MinDifferentiability<F, G>::value;
  return ProdExpression<F, G, TMode>(f, g);
}

// Unary minus: returns a MulExpression representing (-1)*f.
template <typename F>
auto operator-(const F& f)
    -> MulExpression<F, typename F::ScalarType, F::Differentiability> {
  return (-1) * f;
}

// Overload for adding a constant scalar to a function (f + c).
template <typename F>
auto operator+(const F& f, const typename F::ScalarType& c) -> AddExpression<
    F,
    ConstExpression<typename F::ScalarType, F::Differentiability, F::Dimension>,
    F::Differentiability> {
  using ScalarType = typename F::ScalarType;
  ConstExpression<ScalarType, F::Differentiability, F::Dimension> c_expr(c);
  return AddExpression<
      F, ConstExpression<ScalarType, F::Differentiability, F::Dimension>,
      F::Differentiability>(f, c_expr);
}

// Overload for adding a function to a constant scalar (c + f).
template <typename F>
auto operator+(const typename F::ScalarType& c, const F& f) -> AddExpression<
    ConstExpression<typename F::ScalarType, F::Differentiability, F::Dimension>,
    F, F::Differentiability> {
  using ScalarType = typename F::ScalarType;
  ConstExpression<ScalarType, F::Differentiability, F::Dimension> c_expr(c);
  return AddExpression<
      ConstExpression<ScalarType, F::Differentiability, F::Dimension>, F,
      F::Differentiability>(c_expr, f);
}

// Overload for subtracting a constant scalar from a function (f - c).
template <typename F>
auto operator-(const F& f, const typename F::ScalarType& c) {
  using ScalarType = typename F::ScalarType;
  // Create a constant expression for the scalar.
  ConstExpression<ScalarType, F::Differentiability, F::Dimension> c_expr(c);
  // Combine f and c_expr using SubExpression.
  return SubExpression<
      F, ConstExpression<ScalarType, F::Differentiability, F::Dimension>,
      F::Differentiability>(f, c_expr);
}

// Overload for subtracting a function from a constant scalar (c - f).
template <typename F>
auto operator-(const typename F::ScalarType& c, const F& f) {
  using ScalarType = typename F::ScalarType;
  // Create a constant expression for the scalar.
  ConstExpression<ScalarType, F::Differentiability, F::Dimension> c_expr(c);
  // Combine c_expr and f using SubExpression.
  return SubExpression<
      ConstExpression<ScalarType, F::Differentiability, F::Dimension>, F,
      F::Differentiability>(c_expr, f);
}

}  // namespace cppoptlib::function

#endif  //  INCLUDE_CPPOPTLIB_FUNCTION_EXPRESSIONS_H_
