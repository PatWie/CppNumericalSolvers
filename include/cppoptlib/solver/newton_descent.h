// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
#define INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_

#include <iostream>

#include "../linesearch/armijo.h"
#include "Eigen/Dense"
#include "solver.h" // NOLINT

namespace cppoptlib::solver {

template <typename FunctionType>
class NewtonDescent
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(FunctionType::Differentiability ==
                    cppoptlib::function::DifferentiabilityMode::Second,
                "NewtonDescent only supports second-order "
                "differentiable functions");

private:
  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using progress_t = typename Superclass::progress_t;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using MatrixType = typename FunctionType::MatrixType;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void InitializeSolver(const FunctionType & /*function*/,
                        const StateType &initial_state) override {
    dim_ = initial_state.x.rows();
  }

  StateType OptimizationStep(const FunctionType &function,
                             const StateType &current,
                             const progress_t & /*state*/) override {
    constexpr ScalarType safe_guard = 1e-5;

    MatrixType hessian;
    VectorType gradient;
    function(current.x, &gradient, &hessian);
    hessian += safe_guard * MatrixType::Identity(dim_, dim_);

    const VectorType delta_x = hessian.lu().solve(-gradient);
    const ScalarType rate = linesearch::Armijo<FunctionType, 2>::Search(
        current.x, delta_x, function);

    return StateType(current.x + rate * delta_x);
  }

private:
  int dim_;
};

} // namespace cppoptlib::solver

#endif // INCLUDE_CPPOPTLIB_SOLVER_NEWTON_DESCENT_H_
