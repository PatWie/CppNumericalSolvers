// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#include <iostream>

#include "include/cppoptlib/function.h"
#include "include/cppoptlib/solver/gradient_descent.h"

template <typename T>
class Simple : public cppoptlib::function::Function<T, 2> {
 public:
  using Vector = typename cppoptlib::function::Function<T, 2>::Vector;
  using Scalar = typename cppoptlib::function::Function<T, 2>::Scalar;

  Scalar operator()(const Vector &x) const override {
    return 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;
  }

  void gradient(const Vector &x, Vector *grad) const {
    (*grad)[0] = 2 * 5 * x[0];
    (*grad)[1] = 2 * 100 * x[1];
  }
};

int main(int argc, char const *argv[]) {
  using Function = Simple<double>;
  Function f;
  Function::Vector x(2);
  x << -1, 2;

  Function::Vector dx;
  Function::Hessian h;

  f.gradient(x, &dx);
  f.hessian(x, &h);

  std::cout << f(x) << std::endl;
  std::cout << dx << std::endl;
  std::cout << h << std::endl;

  std::cout << cppoptlib::utils::IsGradientCorrect(f, x) << std::endl;
  std::cout << cppoptlib::utils::IsHessianCorrect(f, x) << std::endl;

  cppoptlib::solver::GradientDescent<Function> solver;
  solver.minimize(f, &x);
  std::cout << "argmin " << x.transpose() << std::endl;
  std::cout << "f in argmin " << f(x) << std::endl;
  std::cout << "solver state " << solver.CurrentState().num_iterations
            << std::endl;

  return 0;
}
