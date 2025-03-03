// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_
#define INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_

namespace cppoptlib::solver::linesearch {
template <typename function_t, int Ord>
class Armijo {
 public:
  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  /**
   * @brief use Armijo Rule for (weak) Wolfe conditiions
   * @details [long description]
   *
   * @param search_direction search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */
  static scalar_t Search(const vector_t &x, const vector_t &search_direction,
                         const function_t &function,
                         const scalar_t alpha_init = 1.0) {
    constexpr scalar_t c = 0.2;
    constexpr scalar_t rho = 0.9;
    scalar_t alpha = alpha_init;
    vector_t gradient;
    const scalar_t f_in = function(x, &gradient);
    scalar_t f = function(x + alpha * search_direction);
    const scalar_t cache = c * gradient.dot(search_direction);

    while (f > f_in + alpha * cache) {
      alpha *= rho;
      f = function(x + alpha * search_direction);
    }

    return alpha;
  }
};

template <typename function_t>
class Armijo<function_t, 2> {
 public:
  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  using matrix_t = typename function_t::matrix_t;
  /**
   * @brief use Armijo Rule for (weak) Wolfe conditiions
   * @details [long description]
   *
   * @param search_direction search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */
  static scalar_t Search(const vector_t &x, const vector_t &search_direction,
                         const function_t &function) {
    constexpr scalar_t c = 0.2;
    constexpr scalar_t rho = 0.9;
    scalar_t alpha = 1.0;
    vector_t gradient;
    matrix_t hessian;
    const scalar_t f_in = function(x, &gradient, &hessian);
    scalar_t f = function(x + alpha * search_direction);
    const scalar_t cache =
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
