# CppNumericalSolvers: A Modern C++17 Header-Only Optimization Library

CppNumericalSolvers is a lightweight, header-only C++17 library for numerical
optimization. It provides a suite of modern, efficient solvers for both
unconstrained and constrained problems, designed for easy integration and high
performance. A key feature is its use of function expression templates, which
allow you to build complex objective functions on-the-fly without writing
boilerplate code.

The library is built on Eigen3 and is distributed under the permissive MIT
license, making it suitable for both academic and commercial projects.

## Core Features

- **Header-Only & Easy Integration**: Simply include the headers in your
    project. No complex build steps required.
- **Modern C++17 Design**: Utilizes modern C++ features for a clean, type-safe,
    and expressive API.
- **Powerful Expression Templates**: Compose complex objective functions from
    simpler parts using operator overloading (`+`, `-`, `*`). This avoids manual
    implementation of wrapper classes and promotes reusable code.
- **Comprehensive Solver Suite**:
  - **Unconstrained Solvers**: Gradient Descent, Conjugate Gradient, Newton's
      Method, BFGS, L-BFGS, and Nelder-Mead.
  - **Constrained Solvers**: L-BFGS-B (for box constraints) and the Augmented
      Lagrangian method (for general equality and inequality constraints).
- **Automatic Differentiation Utilities**: Includes tools to verify the
    correctness of your analytical gradients and Hessians using finite difference
    approximations.
- **Permissive MIT License**: Free to use in any project, including commercial
    applications.

## Quick Start: Basic Minimization

Here’s how to solve a simple unconstrained optimization problem. We'll minimize
the function *f(x)=5x₀² + 100x₁² + 5*.

### 1. Define the Objective Function

First, create a class for your objective function that inherits from
`cppoptlib::function::FunctionCRTP`. You need to implement `operator()` which
returns the function value and optionally computes the gradient and Hessian.

```cpp
#include <iostream>
#include "cppoptlib/function.h"
#include "cppoptlib/solver/lbfgs.h"

// Use a CRTP-based class to define a 2D function with second-order derivatives.
class MyObjective : public cppoptlib::function::FunctionCRTP<MyObjective, double, cppoptlib::function::DifferentiabilityMode::Second, 2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // The objective function: f(x) = 5*x₀² + 100*x₁² + 5
  ScalarType operator()(const VectorType &x, VectorType *gradient, MatrixType *hessian) const {
    if (gradient) {
      (*gradient)(0) = 10 * x(0);
      (*gradient)(1) = 200 * x(1);
    }
    if (hessian) {
      (*hessian)(0, 0) = 10;
      (*hessian)(0, 1) = 0;
      (*hessian)(1, 0) = 0;
      (*hessian)(1, 1) = 200;
    }
    return 5 * x(0) * x(0) + 100 * x(1) * x(1) + 5;
  }
};
```

### 2. Solve the Problem

Instantiate your function, pick a solver, set a starting point, and run the minimization.

```cpp
int main() {
  MyObjective f;
  Eigen::Vector2d x_init;
  x_init << -10, 2;

  // Choose a solver (L-BFGS is a great all-rounder)
  cppoptlib::solver::Lbfgs<MyObjective> solver;

  // Create the initial state for the solver
  auto initial_state = cppoptlib::function::FunctionState(x_init);

  // Set a callback to print progress
  solver.SetCallback(cppoptlib::solver::PrintProgressCallback<MyObjective, decltype(initial_state)>(std::cout));

  // Run the minimization
  auto [solution, solver_state] = solver.Minimize(f, initial_state);

  std::cout << "\nSolver finished!" << std::endl;
  std::cout << "Final Status: " << solver_state.status << std::endl;
  std::cout << "Found minimum at: " << solution.x.transpose() << std::endl;
  std::cout << "Function value: " << f(solution.x) << std::endl;

  return 0;
}
```

# The Power of Function Expression Templates

Manually creating a new C++ class for every objective function is tedious,
especially when objectives are just different combinations of standard cost
terms. CppNumericalSolvers uses expression templates to let you build complex
objective functions from modular, reusable "cost functions".

Let's demonstrate this with a practical example: **Ridge Regression**. The goal
is to find model parameters, `x`, that minimize a combination of two terms:

- **Data Fidelity**: How well does the model fit the data? We measure this with
    the squared error: ∥Ax−y∥².
- **Regularization**: How complex is the model? We penalize complexity with the
    L2 norm: ∥x∥².

The final objective is a weighted sum: **F(x) = ∥Ax−y∥² + λ∥x∥²**, where `λ` is
a scalar weight that controls the trade-off.

With expression templates, we can define each term as a separate, reusable
class and then combine them with a single line of C++: **DataTerm + lambda *
RegularizationTerm**.

---

## 1. Define the Building Blocks

First, we create classes for our two cost terms. Each is a self-contained, differentiable function.

```cpp
#include <iostream>
#include "cppoptlib/function.h"
#include "cppoptlib/solver/lbfgs.h"

// The first term: Data Fidelity as Squared Error: f(x) = ||Ax - y||^2
class SquaredError : public cppoptlib::function::FunctionCRTP<SquaredError, double, cppoptlib::function::DifferentiabilityMode::Second> {
private:
    const Eigen::MatrixXd &A;
    const Eigen::VectorXd &y;

public:
    SquaredError(const Eigen::MatrixXd &A, const Eigen::VectorXd &y) : A(A), y(y) {}

    int GetDimension() const { return A.cols(); }

    ScalarType operator()(const VectorType &x, VectorType *grad, MatrixType *hess) const {
        Eigen::VectorXd residual = A * x - y;
        if (grad) {
            *grad = 2 * A.transpose() * residual;
        }
        if (hess) {
            *hess = 2 * A.transpose() * A;
        }
        return residual.squaredNorm();
    }
};

// The second term: L2 Regularization: g(x) = ||x||^2
class L2Regularization : public cppoptlib::function::FunctionCRTP<L2Regularization, double, cppoptlib::function::DifferentiabilityMode::Second> {
public:
    int dim;
    explicit L2Regularization(int d) : dim(d) {}

    int GetDimension() const { return dim; }

    ScalarType operator()(const VectorType &x, VectorType *grad, MatrixType *hess) const {
        if (grad) {
            *grad = 2 * x;
        }
        if (hess) {
            hess->setIdentity(dim, dim);
            *hess *= 2;
        }
        return x.squaredNorm();
    }
};
```

## 2. Compose and Solve in `main`

Now, we can combine these building blocks to create our final objective function and solve the problem.

```cpp
int main() {
    // 1. Define the problem data
    Eigen::MatrixXd A(3, 2);
    A << 1, 2,
         3, 4,
         5, 6;
    Eigen::VectorXd y(3);
    y << 7, 8, 9;

    const double lambda = 0.1; // Regularization weight

    // 2. Create instances of our reusable cost functions
    auto data_term = SquaredError(A, y);
    auto reg_term = L2Regularization(A.cols());

    // 3. Compose the final objective using expression templates!
    // F(x) = (||Ax - y||^2) + lambda * (||x||^2)
    auto objective_expr = data_term + lambda * reg_term;

    // 4. Wrap the expression for the solver. The library automatically handles
    // the combination of values, gradients, and Hessians.
    cppoptlib::function::FunctionExpr objective(objective_expr);

    // 5. Solve as usual
    Eigen::VectorXd x_init = Eigen::VectorXd::Zero(A.cols());
    cppoptlib::solver::Lbfgs<decltype(objective)> solver;

    auto [solution, solver_state] = solver.Minimize(objective, cppoptlib::function::FunctionState(x_init));

    std::cout << "Found minimum at: " << solution.x.transpose() << std::endl;
    std::cout << "Function value: " << objective(solution.x) << std::endl;
}
```

## Constrained Optimization

Solve constrained problems using the Augmented Lagrangian method. Here, we
solve *min f(x) = x₀ + x₁* subject to the equality constraint *x₀² + x₁² - 2 =
0*. The optimal solution lies on the circle of radius √2 at the point (-1, -1).

```cpp
#include "cppoptlib/function.h"
#include "cppoptlib/solver/augmented_lagrangian.h"
#include "cppoptlib/solver/lbfgs.h"

// Objective: f(x) = x_0 + x_1
class SumObjective : public cppoptlib::function::FunctionXd<SumObjective> {
 public:
  ScalarType operator()(const VectorType &x, VectorType *grad) const {
    if (grad) *grad = VectorType::Ones(x.size());
    return x.sum();
  }
};

// Constraint: c(x) = x_0^2 + x_1^2
class Circle : public cppoptlib::function::FunctionXd<Circle> {
 public:
  ScalarType operator()(const VectorType &x, VectorType *grad) const {
    if (grad) *grad = 2 * x;
    return x.squaredNorm();
  }
};

int main() {
  cppoptlib::function::FunctionExpr objective = SumObjective();
  cppoptlib::function::FunctionExpr circle_constraint_base = Circle();

  cppoptlib::function::FunctionExpr equality_constraint = circle_constraint_base - 2.0;

  cppoptlib::function::ConstrainedOptimizationProblem problem(
      objective, {equality_constraint});

  cppoptlib::solver::Lbfgs<decltype(problem)::ObjectiveFunctionType> inner_solver;
  cppoptlib::solver::AugmentedLagrangian solver(problem, inner_solver);

  Eigen::VectorXd x_init(2);
  x_init << 5, -3;

  cppoptlib::solver::AugmentedLagrangeState<double> state(x_init, 1, 0, 10.0);

  auto [solution, solver_state] = solver.Minimize(problem, state);

  std::cout << "Solver finished!" << std::endl;
  std::cout << "Found minimum at: " << solution.x.transpose() << std::endl;
  std::cout << "Function value: " << objective(solution.x) << std::endl;

  return 0;
}
```

## Installation

CppNumericalSolvers is header-only. You just need a C++17 compatible compiler
and Eigen3.

### Recommended: CMake FetchContent

Add this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  cppoptlib
  GIT_REPOSITORY https://github.com/PatWie/CppNumericalSolvers.git
  GIT_TAG main # Or a specific commit/tag
)

FetchContent_MakeAvailable(cppoptlib)

# ... then in your target:
target_link_libraries(your_target PRIVATE CppNumericalSolvers)
```

### Manual Integration

Clone the repository:

```bash
git clone https://github.com/PatWie/CppNumericalSolvers.git
```

Add the `include/` directory to your project's include path.
Ensure Eigen3 is available in your include path.

## Citation

If you use this library in your research, please cite it:

```bibtex
@misc{wieschollek2016cppoptimizationlibrary,
  title={CppOptimizationLibrary},
  author={Wieschollek, Patrick},
  year={2016},
  howpublished={\url{https://github.com/PatWie/CppNumericalSolvers}},
}
```

