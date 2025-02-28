// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#include <iostream>
#include <limits>

#include "include/cppoptlib/function.h"
#include "include/cppoptlib/solver/bfgs.h"
#include "include/cppoptlib/solver/conjugated_gradient_descent.h"
#include "include/cppoptlib/solver/gradient_descent.h"
#include "include/cppoptlib/solver/lbfgs.h"
#include "include/cppoptlib/solver/lbfgsb.h"
#include "include/cppoptlib/solver/nelder_mead.h"
#include "include/cppoptlib/solver/newton_descent.h"
#include "include/cppoptlib/utils/derivatives.h"

using FunctionXd = cppoptlib::function::Function<
    double, Eigen::Dynamic, cppoptlib::function::Differentiability::Second>;
// Or be more specific.
using Function2d = cppoptlib::function::Function<
    double, 2, cppoptlib::function::Differentiability::First>;

class Function : public FunctionXd {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  state_t operator()(const vector_t &x) const override {
    state_t state;
    state.x = x;

    state.value = 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;

    state.gradient = vector_t::Zero(2);
    state.gradient[0] = 2 * 5 * x[0];
    state.gradient[1] = 2 * 100 * x[1];

    state.hessian = matrix_t::Zero(2, 2);
    state.hessian(0, 0) = 10;
    state.hessian(0, 1) = 0;
    state.hessian(1, 0) = 0;
    state.hessian(1, 1) = 200;

    return state;
  }
};

int main() {
  // using Solver = cppoptlib::solver::GradientDescent<Function>;
  // using Solver = cppoptlib::solver::ConjugatedGradientDescent<Function>;
  // using Solver = cppoptlib::solver::NewtonDescent<Function>;
  // using Solver = cppoptlib::solver::Bfgs<Function>;
  using Solver = cppoptlib::solver::Lbfgs<Function>;
  // using Solver = cppoptlib::solver::Lbfgsb<Function>;
  // using Solver = cppoptlib::solver::NelderMead<Function>;

  constexpr auto dim = 2;
  Function f;
  Function::vector_t x(dim);
  x << -1, 2;

  Function::state_t init_state = f(x);

  std::cout << init_state.value << std::endl;
  std::cout << init_state.gradient << std::endl;

  std::cout << cppoptlib::utils::IsGradientCorrect(f, x) << std::endl;
  std::cout << cppoptlib::utils::IsHessianCorrect(f, x) << std::endl;

  // Solver solver;
  Solver solver(cppoptlib::solver::DefaultStoppingSolverProgress<Function>(),
                cppoptlib::solver::PrintCallback<Function>());

  auto [solution, solver_state] = solver.Minimize(f, init_state);
  std::cout << "argmin " << solution.x.transpose() << std::endl;
  std::cout << "f in argmin " << solution.value << std::endl;
  std::cout << "iterations " << solver_state.num_iterations << std::endl;
  std::cout << "status " << solver_state.status << std::endl;

  return 0;
}
