# CppNumericalSolvers

A header-only C++17 optimization library that is fast, reliable, and
easy to integrate.

On an [83-problem benchmark][bench] against Nocedal's Fortran L-BFGS,
libLBFGS, LBFGSpp, and LBFGS-Lite, CppNumericalSolvers has the
highest reliability (95% converged), the most first-place wins
(43 / 83, 2× the next library), and the lowest mean nfev of any
solver tested.

[bench]: https://patwie.github.io/CppNumericalSolversBenchmark/

## Quick Start

```cpp
#include "cppoptlib/function.h"
#include "cppoptlib/solver/lbfgs.h"

// f(x) = 5*x0^2 + 100*x1^2 + 5
class Quadratic : public cppoptlib::function::FunctionCRTP<
    Quadratic, double, cppoptlib::function::DifferentiabilityMode::First, 2> {
 public:
  ScalarType operator()(const VectorType &x, VectorType *grad) const {
    if (grad) *grad = Eigen::Vector2d(10 * x[0], 200 * x[1]);
    return 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;
  }
};

int main() {
  Quadratic f;
  Eigen::Vector2d x0(-10, 2);
  cppoptlib::solver::Lbfgs<Quadratic> solver;
  auto [solution, state] = solver.Minimize(f, cppoptlib::function::FunctionState(x0));
  // solution.x ≈ (0, 0), solution.value ≈ 5
}
```

Add to your project via CMake FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(cppoptlib
  GIT_REPOSITORY https://github.com/PatWie/CppNumericalSolvers.git
  GIT_TAG main)
FetchContent_MakeAvailable(cppoptlib)
target_link_libraries(your_target PRIVATE CppNumericalSolvers)
```

Or clone and add `include/` to your include path.  The only dependency
is Eigen3.

## Solvers

| Solver | Header | Order | Constraints |
|---|---|---|---|
| Gradient Descent | `solver/gradient_descent.h` | 1st | — |
| Conjugate Gradient | `solver/conjugated_gradient_descent.h` | 1st | — |
| L-BFGS | `solver/lbfgs.h` | 1st | — |
| BFGS | `solver/bfgs.h` | 1st | — |
| Newton | `solver/newton_descent.h` | 2nd | — |
| Trust-Region Newton | `solver/trust_region_newton.h` | 2nd | — |
| Nelder-Mead | `solver/nelder_mead.h` | 0th | — |
| L-BFGS-B | `solver/lbfgsb.h` | 1st | box |
| Augmented Lagrangian | `solver/augmented_lagrangian.h` | any | equality / inequality |

## Expression Templates

Build complex objectives from reusable parts without boilerplate.
Example: Ridge Regression *F(x) = ||Ax - y||² + λ||x||²*.

```cpp
#include "cppoptlib/function.h"
#include "cppoptlib/solver/lbfgs.h"

class SquaredError : public cppoptlib::function::FunctionCRTP<
    SquaredError, double, cppoptlib::function::DifferentiabilityMode::Second> {
  const Eigen::MatrixXd &A;
  const Eigen::VectorXd &y;
 public:
  SquaredError(const Eigen::MatrixXd &A, const Eigen::VectorXd &y) : A(A), y(y) {}
  int GetDimension() const { return A.cols(); }
  ScalarType operator()(const VectorType &x, VectorType *grad, MatrixType *hess) const {
    Eigen::VectorXd r = A * x - y;
    if (grad) *grad = 2 * A.transpose() * r;
    if (hess) *hess = 2 * A.transpose() * A;
    return r.squaredNorm();
  }
};

class L2Reg : public cppoptlib::function::FunctionCRTP<
    L2Reg, double, cppoptlib::function::DifferentiabilityMode::Second> {
  int dim;
 public:
  explicit L2Reg(int d) : dim(d) {}
  int GetDimension() const { return dim; }
  ScalarType operator()(const VectorType &x, VectorType *grad, MatrixType *hess) const {
    if (grad) *grad = 2 * x;
    if (hess) { hess->setIdentity(dim, dim); *hess *= 2; }
    return x.squaredNorm();
  }
};

int main() {
  Eigen::MatrixXd A(3, 2);  A << 1,2, 3,4, 5,6;
  Eigen::VectorXd y(3);     y << 7, 8, 9;
  double lambda = 0.1;

  // Compose: F(x) = ||Ax-y||^2 + 0.1 * ||x||^2
  cppoptlib::function::FunctionExpr objective(SquaredError(A, y) + lambda * L2Reg(A.cols()));

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(A.cols());
  cppoptlib::solver::Lbfgs<decltype(objective)> solver;
  auto [sol, state] = solver.Minimize(objective, cppoptlib::function::FunctionState(x0));
  std::cout << "x* = " << sol.x.transpose() << ", f* = " << sol.value << "\n";
}
```

## Constrained Optimization

Solve *min x₀ + x₁* subject to *x₀² + x₁² = 2* using the Augmented
Lagrangian method:

```cpp
#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/lbfgs.h"

class Sum : public cppoptlib::function::FunctionXd<Sum> {
 public:
  ScalarType operator()(const VectorType &x, VectorType *g) const {
    if (g) *g = VectorType::Ones(x.size());
    return x.sum();
  }
};

class CircleNorm : public cppoptlib::function::FunctionXd<CircleNorm> {
 public:
  ScalarType operator()(const VectorType &x, VectorType *g) const {
    if (g) *g = 2 * x;
    return x.squaredNorm();
  }
};

int main() {
  cppoptlib::function::FunctionExpr objective = Sum();
  cppoptlib::function::FunctionExpr constraint = cppoptlib::function::FunctionExpr(CircleNorm()) - 2.0;

  cppoptlib::function::ConstrainedOptimizationProblem problem(objective, {constraint});

  cppoptlib::solver::Lbfgs<decltype(problem)::ObjectiveFunctionType> inner;
  cppoptlib::solver::AugmentedLagrangian solver(problem, inner);

  Eigen::VectorXd x0(2);  x0 << 5, -3;
  auto [sol, state] = solver.Minimize(problem,
      cppoptlib::solver::AugmentedLagrangeState<double>(x0, 1, 0, 10.0));
  // sol.x ≈ (-1, -1), f* ≈ -2
}
```

## Defining Functions

Inherit from `FunctionCRTP` and implement `operator()`.  The template
parameters control scalar type, differentiability order, and
(optionally) compile-time dimension.

```cpp
// Dynamic-dimension, first-order:
class MyFunc : public cppoptlib::function::FunctionCRTP<
    MyFunc, double, cppoptlib::function::DifferentiabilityMode::First> { ... };

// Fixed 3D, second-order:
class My3D : public cppoptlib::function::FunctionCRTP<
    My3D, double, cppoptlib::function::DifferentiabilityMode::Second, 3> { ... };
```

Use `cppoptlib::utils::IsGradientCorrect` and `IsHessianCorrect` to
verify your derivatives against finite differences during development.

## Benchmark

Full reproducible benchmark with driver sources, per-iteration
convergence traces, interactive performance profiles, and
per-problem side-by-side results:
**[View results](https://patwie.github.io/CppNumericalSolversBenchmark/)** ·
[Source](https://github.com/PatWie/CppNumericalSolversBenchmark).

## Citation

```bibtex
@misc{wieschollek2016cppoptimizationlibrary,
  title={CppOptimizationLibrary},
  author={Wieschollek, Patrick},
  year={2016},
  howpublished={\url{https://github.com/PatWie/CppNumericalSolvers}},
}
```

## License

MIT.  See [LICENSE](LICENSE) for details.
