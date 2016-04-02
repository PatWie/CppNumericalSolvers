// CppNumericalSolver
#ifndef ISOLVER_H_
#define ISOLVER_H_

#include <functional>
#include "../meta.h"
#include "../problem.h"

namespace cppoptlib {

template<typename T, int Ord>
class ISolver {
public:
    class Info {
    public:
        T gradNorm;
        T rate;
        size_t iterations;
        size_t m;
    };

protected:
  const int order_ = Ord;
  Info m_ctrl;
  Info m_info;

  virtual bool shouldContinue() {
      return (this->m_info.gradNorm > this->m_ctrl.gradNorm) && (this->m_info.iterations < this->m_ctrl.iterations);
  }

public:
  ISolver() :
    m_ctrl(),
    m_info()
  {
    m_ctrl.rate = 0.00005;
    m_ctrl.iterations = 100000;
    m_ctrl.gradNorm = 1e-4;
    m_ctrl.m = 10;
  }

  ISolver(const Info &control) :
    m_ctrl(control),
    m_info()
  {}

  /**
   * @brief minimize an objective function given a gradient (and optinal a hessian)
   * @details this is just the abstract interface
   *
   * @param x0 starting point
   * @param funObjective objective function
   * @param funGradient gradient function
   * @param funcHession hessian function
   */
  virtual void minimize(Problem<T> &objFunc, Vector<T> & x0) = 0;

  /*
   * @brief Let the user set the control information structure.
   */
  Info &ctrl() { return m_ctrl; }

  /*
   * @brief Let the user query the information structure.
   */
   const Info &info() { return m_info; }

};

} /* namespace cppoptlib */

#endif /* ISOLVER_H_ */
