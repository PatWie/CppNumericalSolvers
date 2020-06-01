// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_
#define INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_

namespace cppoptlib {
namespace solver {
namespace linesearch {
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
  static scalar_t search(const vector_t &x, const vector_t &search_direction,
                         const function_t &function,
                         const scalar_t alpha_init = 1.0) {
    const scalar_t c = 0.2;
    const scalar_t rho = 0.9;
    scalar_t alpha = alpha_init;
    scalar_t f = function(x + alpha * search_direction);
    const scalar_t f_in = function(x);
    vector_t grad(x.rows());
    function.Gradient(x, grad);
    const scalar_t Cache = c * grad.dot(search_direction);

    while (f > f_in + alpha * Cache) {
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
  using hessian_t = typename function_t::hessian_t;
  /**
   * @brief use Armijo Rule for (weak) Wolfe conditiions
   * @details [long description]
   *
   * @param search_direction search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */
  static scalar_t search(const vector_t &x, const vector_t &search_direction,
                         const function_t &function) {
    const scalar_t c = 0.2;
    const scalar_t rho = 0.9;
    scalar_t alpha = 1.0;

    scalar_t f = function(x + alpha * search_direction);
    const scalar_t f_in = function(x);
    hessian_t hessian(x.rows(), x.rows());
    function.Hessian(x, &hessian);
    vector_t grad(x.rows());
    function.Gradient(x, &grad);
    const scalar_t Cache = c * grad.dot(search_direction) +
                           0.5 * c * c * search_direction.transpose() *
                               (hessian * search_direction);

    while (f > f_in + alpha * Cache) {
      alpha *= rho;
      f = function(x + alpha * search_direction);
    }
    return alpha;
  }
};

};  // namespace linesearch
};  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_