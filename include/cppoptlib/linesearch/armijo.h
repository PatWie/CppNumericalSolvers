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
#ifndef INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_
#define INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_

namespace cppoptlib::solver::linesearch {
template <typename FunctionType, int Ord>
class Armijo {
 public:
  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  /**
   * @brief use Armijo Rule for (weak) Wolfe conditiions
   * @details [long description]
   *
   * @param search_direction search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */
  static ScalarType Search(const VectorType &x,
                           const VectorType &search_direction,
                           const FunctionType &function,
                           const ScalarType alpha_init = 1.0) {
    constexpr ScalarType c = 0.2;
    constexpr ScalarType rho = 0.9;
    ScalarType alpha = alpha_init;
    VectorType gradient;
    const ScalarType f_in = function(x, &gradient);
    ScalarType f = function(x + alpha * search_direction);
    const ScalarType cache = c * gradient.dot(search_direction);
    constexpr ScalarType alpha_min = ScalarType{1e-8};

    while (f > f_in + alpha * cache && alpha > alpha_min) {
      alpha *= rho;
      f = function(x + alpha * search_direction);
    }

    return alpha;
  }
};

template <typename FunctionType>
class Armijo<FunctionType, 2> {
 public:
  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using MatrixType = typename FunctionType::MatrixType;
  /**
   * @brief use Armijo Rule for (weak) Wolfe conditiions
   * @details [long description]
   *
   * @param search_direction search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */
  static ScalarType Search(const VectorType &x,
                           const VectorType &search_direction,
                           const FunctionType &function) {
    constexpr ScalarType c = 0.2;
    constexpr ScalarType rho = 0.9;
    ScalarType alpha = 1.0;
    VectorType gradient;
    MatrixType hessian;
    const ScalarType f_in = function(x, &gradient, &hessian);
    ScalarType f = function(x + alpha * search_direction);
    const ScalarType cache =
        c * gradient.dot(search_direction) +
        0.5 * c * c * search_direction.transpose() * hessian * search_direction;

    while (f > f_in + alpha * cache) {
      alpha *= rho;
      f = function(x + alpha * search_direction);
    }
    return alpha;
  }
};
}  // namespace cppoptlib::solver::linesearch

#endif  // INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_
