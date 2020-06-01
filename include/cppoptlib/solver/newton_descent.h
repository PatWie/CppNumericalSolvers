// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_

#include "../linesearch/armijo.h"
#include "Eigen/Dense"
#include "solver.h"

namespace cppoptlib {

namespace solver {
template <typename Function>
class NewtonDescent : public Solver<Function, 2> {
 public:
  using Superclass = Solver<Function, 2>;
  using typename Superclass::Dim;
  using typename Superclass::ScalarT;
  using typename Superclass::VectorT;
  using typename Superclass::HessianT;
  using typename Superclass::FunctionStateT;

  FunctionStateT optimization_step(const Function &function,
                                   const FunctionStateT &state) override {
    FunctionStateT current(state);

    std::cout << "------- >" << state.value << std::endl;
    std::cout << "this" << std::endl;

    std::cout << current.gradient << std::endl;
    std::cout << "this" << std::endl;


    constexpr ScalarT safe_guard = 1e-5;
    current.hessian += safe_guard * HessianT::Identity();

    const VectorT delta_x = current.hessian.lu().solve(-current.gradient);
    std::cout << delta_x.transpose() << std::endl;


    const ScalarT rate = linesearch::Armijo<Function, 2>::search(
        current.x, delta_x, function);

    current.x = current.x - rate * delta_x;
    current.value = function(current.x);
    std::cout << "------- >" << current.value << std::endl;
    function.Gradient(current.x, &current.gradient);
    function.Hessian(current.x, &current.hessian);

    return current;
  }
};

};  // namespace solver
} /* namespace cppoptlib */

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_