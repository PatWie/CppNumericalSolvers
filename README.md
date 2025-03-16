# CppNumericalSolvers (A header-only C++17 optimization library)

CppNumericalSolvers is a header-only C++17 optimization library providing a
suite of solvers for both unconstrained and constrained optimization problems.
The library is designed for efficiency, modern C++ compliance, and easy
integration into existing projects. It is distributed under a permissive
license, making it suitable for commercial use.

Key Features:
- **Unconstrained Optimization**: Supports algorithms such as **Gradient Descent, Conjugate Gradient, Newton's Method, BFGS, L-BFGS, and Nelder-Mead**.
- **Constrained Optimization**: Provides support for **L-BFGS-B** and **Augmented Lagrangian methods**.
- **Expression Templates**: Enable efficient evaluation of function expressions without unnecessary allocations.
- **First- and Second-Order Methods**: Support for both **gradient-based** and **Hessian-based** optimizations.
- **Differentiation Utilities**: Tools to check gradient and Hessian correctness.
- **Header-Only**: Easily integrate into any C++17 project with no dependencies.


## Quick Start

### **Minimizing a Function (BFGS Solver)**

```cpp
#include <iostream>
#include "cppoptlib/function.h"
#include "cppoptlib/solver/bfgs.h"

using namespace cppoptlib::function;

class ExampleFunction : public FunctionXd<ExampleFunction> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * The function is defined as:
     *
     *     f(x) = 5 * x₀² + 100 * x₁² + 5
     *
     * Its gradient is given by:
     *
     *     ∇f(x) = [10 * x₀, 200 * x₁]ᵀ
     *
     * And its Hessian is:
     *
     *     ∇²f(x) = [ [10,   0],
     *                 [ 0, 200] ]
     */
  ScalarType operator()(const VectorType &x, VectorType *gradient = nullptr,
                        MatrixType *hessian = nullptr) const override {
    if (gradient) {
      *gradient = VectorType::Zero(2);
      (*gradient)[0] = 2 * 5 * x[0];
      (*gradient)[1] = 2 * 100 * x[1];
    }

    if (hessian) {
      *hessian = MatrixType::Zero(2, 2);
      (*hessian)(0, 0) = 10;
      (*hessian)(0, 1) = 0;
      (*hessian)(1, 0) = 0;
      (*hessian)(1, 1) = 200;
    }

    return 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;
  }
};

int main() {
    ExampleFunction f;
    Eigen::VectorXd x(2);
    x << -1, 2;

    using Solver = cppoptlib::solver::Bfgs<ExampleFunction>;
    auto init_state = f.GetState(x);
    Solver solver;
    auto [solution_state, solver_progress] = solver.Minimize(f, init_state);

    std::cout << "Optimal solution: " << solution_state.x.transpose() << std::endl;
    return 0;
}
```

### **Solving a Constrained Optimization Problem**

CppNumericalSolvers allows solving constrained optimization problems using the
**Augmented Lagrangian** method. Below, we solve the problem:

```
min f(x) = x_0 + x_1
```

subject to the constraint:

```
x_0^2 + x_1^2 = 2
```

```cpp
#include <iostream>
#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/lbfgs.h"

using namespace cppoptlib::function;
using namespace cppoptlib::solver;

class SumObjective : public FunctionXd<SumObjective> {
 public:
  ScalarType operator()(const VectorType &x, VectorType *gradient = nullptr) const {
    if (gradient) *gradient = VectorType::Ones(2);
    return x.sum();
  }
};

class Circle : public FunctionXd<Circle> {
 public:
  ScalarType operator()(const VectorType &x, VectorType *gradient = nullptr) const {
    if (gradient) *gradient = 2 * x;
    return x.squaredNorm();
  }
};

int main() {
  SumObjective::VectorType x(2);
  x << 2, 10;

  FunctionExpr objective = SumObjective();
  FunctionExpr circle = Circle();

  ConstrainedOptimizationProblem prob(
      objective,
      {FunctionExpr(circle - 2)},
      {FunctionExpr(2 - circle)});

  Lbfgs<FunctionExpr2d1> inner_solver;
  AugmentedLagrangian solver(prob, inner_solver);

  AugmentedLagrangeState<double> l_state(x, /* num_eq=*/1,
                                            /* num_ineq=*/1,
                                            /* penalty=*/1.0);
  auto [solution, solver_state] = solver.Minimize(prob, l_state);

  std::cout << "Optimal x: " << solution.x.transpose() << std::endl;
  return 0;
}

```


## Use in Your Project

Either use Bazel or CMake:

```sh
git clone https://github.com/PatWie/CppNumericalSolvers.git
```

Compile with:

```sh
g++ -std=c++17 -Ipath/to/CppNumericalSolvers/include myfile.cpp -o myprogram
```




### Citing this implementation

If you find this implementation useful and wish to cite it, please use the following bibtex entry:

```bibtex
@misc{wieschollek2016cppoptimizationlibrary,
  title={CppOptimizationLibrary},
  author={Wieschollek, Patrick},
  year={2016},
  howpublished={\url{https://github.com/PatWie/CppNumericalSolvers}},
}
```
