// CppNumericalSolver
#ifndef ISOLVER_H_
#define ISOLVER_H_

#include <functional>
#include "isolver.h"
#include "../meta.h"

namespace cns {

template<typename T, int Ord>
class ISolver {
protected:
  const int order_ = Ord;
 public:
  Options settings_;

  ISolver() {
    settings_ = Options();
  }

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

};

} /* namespace cns */

#endif /* ISOLVER_H_ */