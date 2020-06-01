// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers

#include <iostream>

#include "include/cppoptlib/function.h"
#include "include/cppoptlib/solver/gradient_descent.h"

template <typename T>
class Simple : public cppoptlib::function::Function<T, 1, 2> {
 public:
  using VectorT = typename cppoptlib::function::Function<T, 1, 2>::VectorT;
  using ScalarT = typename cppoptlib::function::Function<T, 1, 2>::ScalarT;

  ScalarT operator()(const VectorT &x) const override {
    return 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;
  }

  void Gradient(const VectorT &x, VectorT *grad) const override {
    (*grad)[0] = 2 * 5 * x[0];
    (*grad)[1] = 2 * 100 * x[1];
  }
};

int main(int argc, char const *argv[]) {
  using Function = Simple<double>;
  Function f;
  Function::VectorT x(2);
  x << -1, 2;

  Function::VectorT dx;
  Function::HessianT h;

  f.Gradient(x, &dx);
  f.Hessian(x, &h);

  std::cout << f(x) << std::endl;
  std::cout << dx << std::endl;
  std::cout << h << std::endl;

  std::cout << cppoptlib::utils::IsGradientCorrect(f, x) << std::endl;
  std::cout << cppoptlib::utils::IsHessianCorrect(f, x) << std::endl;

  cppoptlib::solver::GradientDescent<Function> solver;

  auto solution = solver.minimize(f, x);
  std::cout << "argmin " << solution.x.transpose() << std::endl;
  std::cout << "f in argmin " << solution.value << std::endl;
  std::cout << "solver state " << solver.CurrentState().num_iterations
            << std::endl;

  return 0;
}
