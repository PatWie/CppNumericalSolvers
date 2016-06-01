// CppNumericalSolver
#ifndef CONJUGATEDGRADIENTDESCENTSOLVER_H_
#define CONJUGATEDGRADIENTDESCENTSOLVER_H_

#include <Eigen/Dense>
#include "isolver.h"
#include "../linesearch/armijo.h"

namespace cppoptlib {

template<typename T>
class ConjugatedGradientDescentSolver : public ISolverUnbounded<T, 1> {

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

    this->m_current.reset();
    do {
      objFunc.gradient(x0, grad);

      if (this->m_current.iterations == 0) {
        Si = -grad;
      } else {
        const double beta = grad.dot(grad) / (grad_old.dot(grad_old));
        Si = -grad + beta * Si_old;
      }

      const double rate = Armijo<T, decltype(objFunc), 1>::linesearch(x0, Si, objFunc) ;

      x0 = x0 + rate * Si;

      grad_old = grad;
      Si_old = Si;

      this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();
      // std::cout << "iter: "<<iter<< " f = " <<  objFunc.value(x0) << " ||g||_inf "<<gradNorm   << std::endl;
      ++this->m_current.iterations;
      this->m_status = checkConvergence(this->m_stop, this->m_current);
    } while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue) );

  }

};

} /* namespace cppoptlib */

#endif /* CONJUGATEDGRADIENTDESCENTSOLVER_H_ */
