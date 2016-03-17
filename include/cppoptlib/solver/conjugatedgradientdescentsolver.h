// CppNumericalSolver
#ifndef CONJUGATEDGRADIENTDESCENTSOLVER_H_
#define CONJUGATEDGRADIENTDESCENTSOLVER_H_

#include <Eigen/Dense>
#include "isolver.h"
#include "../linesearch/armijo.h"

namespace cppoptlib {

template<typename T>
class ConjugatedGradientDescentSolver : public ISolver<T, 1> {
  using ISolver<T, 1>::ISolver; // Inherit the constructors from the interface
 public:
  /**
   * @brief minimize
   * @details [long description]
   *
   * @param objFunc [description]
   */
  void minimize(Problem<T> &objFunc, Vector<T> & x0) {
    Vector<T> grad(x0.rows());
    Vector<T> grad_old(x0.rows());
    Vector<T> Si(x0.rows());
    Vector<T> Si_old(x0.rows());

    this->m_info = {0., 0., 0, 0}; // Zero out the info struct
    do {
      objFunc.gradient(x0, grad);

      if (this->m_info.iterations == 0) {
        Si = -grad;
      } else {
        const double beta = grad.dot(grad) / (grad_old.dot(grad_old));
        Si = -grad + beta * Si_old;
      }

      const double rate = Armijo<T, decltype(objFunc), 1>::linesearch(x0, Si, objFunc) ;

      x0 = x0 + rate * Si;

      grad_old = grad;
      Si_old = Si;

      this->m_info.gradNorm = grad.template lpNorm<Eigen::Infinity>();
      // std::cout << "Iteration: "<<this->iterations_<< " f = " <<  objFunc.value(x0) << " ||g||_inf "<<gradNorm   << std::endl;
      ++this->m_info.iterations;
    } while (this->shouldContinue());

  }

};

} /* namespace cppoptlib */

#endif /* CONJUGATEDGRADIENTDESCENTSOLVER_H_ */
