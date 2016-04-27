// CppNumericalSolver
#ifndef GRADIENTDESCENTSOLVER_H_
#define GRADIENTDESCENTSOLVER_H_

#include <Eigen/Dense>
#include "isolver.h"
#include "../linesearch/morethuente.h"

namespace cppoptlib {

template<typename T>
class GradientDescentSolver : public ISolver<T, 1> {

public:
  /**
   * @brief minimize
   * @details [long description]
   *
   * @param objFunc [description]
   */
  void minimize(Problem<T> &objFunc, Vector<T> & x0) {

    Vector<T> direction(x0.rows());
    this->m_current.reset();
    do {
      objFunc.gradient(x0, direction);
      const T rate = MoreThuente<T, decltype(objFunc), 1>::linesearch(x0, -direction, objFunc) ;
      x0 = x0 - rate * direction;
      this->m_current.gradNorm = direction.template lpNorm<Eigen::Infinity>();
      // std::cout << "iter: "<<iter<< " f = " <<  objFunc.value(x0) << " ||g||_inf "<<gradNorm  << std::endl;
      ++this->m_current.iterations;
      this->m_status = checkConvergence(this->m_stop, this->m_current);
    } while (this->m_status == Status::Continue);
    if (this->m_debug > DebugLevel::None) {
        std::cout << "Stop status was: " << this->m_status << std::endl;
        std::cout << "Stop criteria were: " << std::endl << this->m_stop << std::endl;
        std::cout << "Current values are: " << std::endl << this->m_current << std::endl;
    }
  }

};

} /* namespace cppoptlib */

#endif /* GRADIENTDESCENTSOLVER_H_ */
