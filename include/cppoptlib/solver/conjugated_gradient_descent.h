// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_

#include <utility>

#include "../linesearch/armijo.h"
#include "Eigen/Core"
#include "solver.h" // NOLINT

namespace cppoptlib::solver {
template <typename FunctionType>
class ConjugatedGradientDescent
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(
      FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::First ||
          FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::Second,
      "ConjugatedGradientDescent only supports first- or second-order "
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

  void InitializeSolver(const FunctionType &function,
                        const StateType &initial_state) override {
    function(initial_state.x, &previous_gradient_);
  }

  StateType OptimizationStep(const FunctionType &function,
                             const StateType &current,
                             const progress_t &progress) override {

    VectorType current_gradient;
    function(current.x, &current_gradient);
    if (progress.num_iterations == 0) {
      search_direction_ = -current_gradient;
    } else {
      const double beta = current_gradient.dot(current_gradient) /
                          (previous_gradient_.dot(previous_gradient_));
      search_direction_ = -current_gradient + beta * search_direction_;
    }
    previous_gradient_ = current_gradient;

    const ScalarType rate = linesearch::Armijo<FunctionType, 1>::Search(
        current.x, search_direction_, function);

    return StateType(current.x + rate * search_direction_);
  }

private:
  VectorType previous_gradient_;
  VectorType search_direction_;
};

} // namespace cppoptlib::solver

#endif // INCLUDE_CPPOPTLIB_SOLVER_CONJUGATED_GRADIENT_DESCENT_H_
