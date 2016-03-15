// CppNumericalSolver
#ifndef GRADIENTDESCENTSOLVER_H_
#define GRADIENTDESCENTSOLVER_H_

#include <Eigen/Dense>
#include "isolver.h"
#include "../linesearch/morethuente.h"

namespace cppoptlib {

template<typename T>
class GradientDescentSolver : public ISolver<T, 1> {
  using ISolver<T, 1>::ISolver; // Inherit the constructors from the interface
 public:
  /**
   * @brief minimize
   * @details [long description]
   *
   * @param objFunc [description]
   */
  void minimize(Problem<T> &objFunc, Vector<T> & x0) {

    Vector<T> direction(x0.rows());
    this->m_info = {0., 0., 0, 0}; // Zero out the information before starting
    do {
      objFunc.gradient(x0, direction);
      const T rate = MoreThuente<T, decltype(objFunc), 1>::linesearch(x0, -direction, objFunc) ;
      x0 = x0 - rate * direction;
      this->m_info.gradNorm = direction.template lpNorm<Eigen::Infinity>();
      // std::cout << "Iteration: "<<this->iterations_<< " f = " <<  objFunc.value(x0) << " ||g||_inf "<<gradNorm  << std::endl;
      ++this->m_info.iterations;
    } while (this->shouldContinue());

  }

};

} /* namespace cppoptlib */

#endif /* GRADIENTDESCENTSOLVER_H_ */
