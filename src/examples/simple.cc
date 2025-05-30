// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#include <iostream>
#include <limits>

#include "Eigen/Core"
#include "cppoptlib/function.h"
#include "cppoptlib/solver/bfgs.h"
#include "cppoptlib/solver/conjugated_gradient_descent.h"
#include "cppoptlib/solver/gradient_descent.h"
#include "cppoptlib/solver/lbfgs.h"
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
#include "cppoptlib/solver/lbfgsb.h"
#endif
#include "cppoptlib/solver/nelder_mead.h"
#include "cppoptlib/solver/newton_descent.h"
#include "cppoptlib/utils/derivatives.h"

template <class F>
using FunctionXd = cppoptlib::function::FunctionCRTP<
    F, double, cppoptlib::function::DifferentiabilityMode::Second>;
using FunctionExprXd2 = cppoptlib::function::FunctionExpr<
    double, cppoptlib::function::DifferentiabilityMode::Second>;

class Function : public FunctionXd<Function> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // static constexpr int Dimension = Eigen::Dynamic;
  // static constexpr cppoptlib::function::DifferentiabilityMode
  //     Differentiability = cppoptlib::function::DifferentiabilityMode::Second;
  // using ScalarType = double;

  ScalarType operator()(const VectorType &x, VectorType *gradient = nullptr,
                        MatrixType *hessian = nullptr

  ) const {
    if (gradient) {
      gradient->resize(x.size());
      (*gradient)[0] = 2 * 5 * x[0];
      (*gradient)[1] = 2 * 100 * x[1];
    }

    if (hessian) {
      hessian->resize(x.size(), x.size());
      (*hessian)(0, 0) = 10;
      (*hessian)(0, 1) = 0;
      (*hessian)(1, 0) = 0;
      (*hessian)(1, 1) = 200;
    }

    return 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;
  }
};

int main() {
  // using Solver = cppoptlib::solver::GradientDescent<FunctionExprXd2>;
  // using Solver =
  // cppoptlib::solver::ConjugatedGradientDescent<FunctionExprXd2>; using Solver
  // = cppoptlib::solver::NewtonDescent<FunctionExprXd2>; using Solver =
  // cppoptlib::solver::Bfgs<FunctionExprXd2>;
  using Solver = cppoptlib::solver::Lbfgs<FunctionExprXd2>;
  // using Solver = cppoptlib::solver::Lbfgsb<FunctionExprXd2>;
  // using Solver = cppoptlib::solver::NelderMead<FunctionExprXd2>;

  constexpr auto dim = 2;
  FunctionExprXd2 f = Function();
  Function::VectorType x(dim);
  x << -10, 2;

  Function::VectorType gradient = Function::VectorType::Zero(2);
  const double value = f(x, &gradient);

  std::cout << value << std::endl;
  std::cout << gradient << std::endl;

  std::cout << cppoptlib::utils::IsGradientCorrect(f, x) << std::endl;
  std::cout << cppoptlib::utils::IsHessianCorrect(f, x) << std::endl;

  const auto initial_state = cppoptlib::function::FunctionState(x);
  std::cout << "init " << initial_state.x.transpose() << std::endl;
  Solver solver;

  solver.SetCallback(
      cppoptlib::solver::PrintProgressCallback<Solver::FunctionType,
                                               Solver::StateType>(std::cout));
  auto [solution, solver_state] = solver.Minimize(f, initial_state);
  std::cout << "argmin " << solution.x.transpose() << std::endl;
  std::cout << "f in argmin " << f(solution.x) << std::endl;
  std::cout << "iterations " << solver_state.num_iterations << std::endl;
  std::cout << "status " << solver_state.status << std::endl;

  return 0;
}
