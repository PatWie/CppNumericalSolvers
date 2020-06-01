// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_
#define INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_

namespace cppoptlib {
namespace solver {
namespace linesearch {
template <typename TFunction, int Ord>
class Armijo {
 public:
  using ScalarT = typename TFunction::ScalarT;
  using VectorT = typename TFunction::VectorT;
  /**
   * @brief use Armijo Rule for (weak) Wolfe conditiions
   * @details [long description]
   *
   * @param searchDir search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */
  static ScalarT search(const VectorT &x, const VectorT &searchDir,
                        const TFunction &function,
                        const ScalarT alpha_init = 1.0) {
    const ScalarT c = 0.2;
    const ScalarT rho = 0.9;
    ScalarT alpha = alpha_init;
    ScalarT f = function(x + alpha * searchDir);
    const ScalarT f_in = function(x);
    VectorT grad(x.rows());
    function.Gradient(x, grad);
    const ScalarT Cache = c * grad.dot(searchDir);

    while (f > f_in + alpha * Cache) {
      alpha *= rho;
      f = function(x + alpha * searchDir);
    }

    return alpha;
  }
};

template <typename TFunction>
class Armijo<TFunction, 2> {
 public:
  using ScalarT = typename TFunction::ScalarT;
  using VectorT = typename TFunction::VectorT;
  using HessianT = typename TFunction::HessianT;
  /**
   * @brief use Armijo Rule for (weak) Wolfe conditiions
   * @details [long description]
   *
   * @param searchDir search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */
  static ScalarT search(const VectorT &x, const VectorT &searchDir,
                        const TFunction &function) {
    const ScalarT c = 0.2;
    const ScalarT rho = 0.9;
    ScalarT alpha = 1.0;

    ScalarT f = function(x + alpha * searchDir);
    const ScalarT f_in = function(x);
    HessianT hessian(x.rows(), x.rows());
    function.Hessian(x, &hessian);
    VectorT grad(x.rows());
    function.Gradient(x, &grad);
    const ScalarT Cache =
        c * grad.dot(searchDir) +
        0.5 * c * c * searchDir.transpose() * (hessian * searchDir);

    while (f > f_in + alpha * Cache) {
      alpha *= rho;
      f = function(x + alpha * searchDir);
    }
    return alpha;
  }
};

};  // namespace linesearch
};  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_LINESEARCH_ARMIJO_H_