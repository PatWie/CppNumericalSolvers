// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#ifndef INCLUDE_CPPOPTLIB_SOLVER_GRADIENTDESCENTSOLVER_H_
#define INCLUDE_CPPOPTLIB_SOLVER_GRADIENTDESCENTSOLVER_H_

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib {

namespace solver {
template <typename Function>
class GradientDescent : public Solver<Function, 1> {
 public:
  using Superclass = Solver<Function, 1>;
  using typename Superclass::Scalar;
  using typename Superclass::Vector;

  void step(const Function &function, Vector *x0, const Vector &gradient) {
    const Scalar rate =
        linesearch::MoreThuente<Function, 1>::search(*x0, -gradient, function);
    *x0 = *x0 - rate * gradient;
  }
};

};  // namespace solver
} /* namespace cppoptlib */

#endif  // INCLUDE_CPPOPTLIB_SOLVER_GRADIENTDESCENTSOLVER_H_
