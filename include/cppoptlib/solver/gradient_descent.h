// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_

#include <utility>

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h" // NOLINT

namespace cppoptlib::solver {
template <typename FunctionType>
class GradientDescent
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(FunctionType::Differentiability ==
                        cppoptlib::function::DifferentiabilityMode::First ||
                    FunctionType::Differentiability ==
                        cppoptlib::function::DifferentiabilityMode::Second,
                "GradientDescent only supports first- or second-order "
                "differentiable functions");

private:
  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using progress_t = typename Superclass::progress_t;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;
  void InitializeSolver(const FunctionType & /*function*/,
                        const StateType & /*initial_state*/) override {}

  StateType OptimizationStep(const FunctionType &function,
                             const StateType &current,
                             const progress_t & /*progress*/) override {
    VectorType gradient;
    function(current.x, &gradient);
    const ScalarType rate = linesearch::MoreThuente<FunctionType, 1>::Search(
        current.x, -gradient, function);

    return StateType(current.x - rate * gradient);
  }
};

} // namespace cppoptlib::solver

#endif // INCLUDE_CPPOPTLIB_SOLVER_GRADIENT_DESCENT_H_
