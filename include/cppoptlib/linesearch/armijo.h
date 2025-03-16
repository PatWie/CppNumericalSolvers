// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
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

    while (f > f_in + alpha * cache) {
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
