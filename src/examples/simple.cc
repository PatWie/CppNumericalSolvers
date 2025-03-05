// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#include <iostream>
#include <limits>

#include "Eigen/Core"
#include "include/cppoptlib/function.h"
#include "include/cppoptlib/solver/bfgs.h"
#include "include/cppoptlib/solver/conjugated_gradient_descent.h"
#include "include/cppoptlib/solver/gradient_descent.h"
#include "include/cppoptlib/solver/lbfgs.h"
#include "include/cppoptlib/solver/lbfgsb.h"
#include "include/cppoptlib/solver/nelder_mead.h"
#include "include/cppoptlib/solver/newton_descent.h"
#include "include/cppoptlib/utils/derivatives.h"

template <class T>
using FunctionXd = cppoptlib::function::Function<
    double, Eigen::Dynamic, T, cppoptlib::function::Differentiability::Second>;

class Function : public FunctionXd<Function> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  scalar_t operator()(const vector_t &x, vector_t *gradient = nullptr,
                      matrix_t *hessian = nullptr) const {
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

  constexpr auto dim = 2;
  Function g;
  auto f = 2 * g + g;
  using Solver = cppoptlib::solver::GradientDescent<decltype(f)>;
  // using Solver = cppoptlib::solver::GradientDescent<decltype(f)>;
  // using Solver = cppoptlib::solver::ConjugatedGradientDescent<decltype(f)>;
  // using Solver = cppoptlib::solver::NewtonDescent<decltype(f)>;
  // using Solver = cppoptlib::solver::Bfgs<decltype(f)>;
  // using Solver = cppoptlib::solver::Lbfgs<decltype(f)>;
  // using Solver = cppoptlib::solver::Lbfgsb<decltype(f)>;
  // using Solver = cppoptlib::solver::NelderMead<decltype(f)>;

  Function::vector_t x(dim);
  x << -1, 2;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(2);
  const double value = f(x, &gradient);

  std::cout << value << std::endl;
  std::cout << gradient << std::endl;

  // std::cout << cppoptlib::utils::IsGradientCorrect(f, x) << std::endl;
  // std::cout << cppoptlib::utils::IsHessianCorrect(f, x) << std::endl;

  // Solver solver;
  Solver solver(cppoptlib::solver::DefaultStoppingSolverProgress<decltype(f)>(),
                cppoptlib::solver::PrintCallback<decltype(f)>());

  auto initial_state = f.GetState(x);
  auto [solution, solver_state] = solver.Minimize(f, initial_state);
  std::cout << "argmin " << solution.x.transpose() << std::endl;
  std::cout << "f in argmin " << f(solution.x) << std::endl;
  std::cout << "iterations " << solver_state.num_iterations << std::endl;
  std::cout << "status " << solver_state.status << std::endl;

  return 0;
}
