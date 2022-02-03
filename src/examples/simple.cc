// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#include <iostream>
#include <limits>

#include "include/cppoptlib/function.h"
#include "include/cppoptlib/solver/bfgs.h"
#include "include/cppoptlib/solver/conjugated_gradient_descent.h"
#include "include/cppoptlib/solver/gradient_descent.h"
#include "include/cppoptlib/solver/lbfgs.h"
#include "include/cppoptlib/solver/lbfgsb.h"
#include "include/cppoptlib/solver/newton_descent.h"

using FunctionXd = cppoptlib::function::Function<double>;

class Function : public FunctionXd {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using FunctionXd::hessian_t;
  using FunctionXd::vector_t;

  scalar_t operator()(const vector_t &x) const override {
    return 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;
  }

  void Gradient(const vector_t &x, vector_t *grad) const override {
    (*grad)[0] = 2 * 5 * x[0];
    (*grad)[1] = 2 * 100 * x[1];
  }

  void Hessian(const vector_t &x, hessian_t *hessian) const override {
    (*hessian)(0, 0) = 10;
    (*hessian)(0, 1) = 0;
    (*hessian)(1, 0) = 0;
    (*hessian)(1, 1) = 200;
  }
};

int main(int argc, char const *argv[]) {
  // using Solver = cppoptlib::solver::NewtonDescent<Function>;
  // using Solver = cppoptlib::solver::GradientDescent<Function>;
  // using Solver = cppoptlib::solver::ConjugatedGradientDescent<Function>;
  // using Solver = cppoptlib::solver::Bfgs<Function>;
  // using Solver = cppoptlib::solver::Lbfgs<Function>;
  using Solver = cppoptlib::solver::Lbfgsb<Function>;

  constexpr auto dim = 2;
  Function f;
  Function::vector_t x(dim);
  x << -1, 2;

  auto state = f.Eval(x);
  std::cout << "this" << std::endl;

  std::cout << f(x) << std::endl;
  std::cout << state.gradient << std::endl;
  if (state.hessian) {
    std::cout << *(state.hessian) << std::endl;
  }

  std::cout << cppoptlib::utils::IsGradientCorrect(f, x) << std::endl;
  std::cout << cppoptlib::utils::IsHessianCorrect(f, x) << std::endl;

  Solver solver{
      Function::vector_t::Constant(
          dim, std::numeric_limits<typename Function::scalar_t>::lowest()),
      Function::vector_t::Constant(
          dim, std::numeric_limits<typename Function::scalar_t>::max())};

  auto [solution, solver_state] = solver.Minimize(f, x);
  std::cout << "argmin " << solution.x.transpose() << std::endl;
  std::cout << "f in argmin " << solution.value << std::endl;
  std::cout << "iterations " << solver_state.num_iterations << std::endl;
  std::cout << "status " << solver_state.status << std::endl;

  return 0;
}
