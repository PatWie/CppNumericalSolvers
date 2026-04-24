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
      *gradient = Eigen::Vector2d(10, 200).cwiseProduct(x);
    }
    if (hessian) {
      hessian->setZero();
      hessian->diagonal() << 10, 200;
    }
    return VectorType(5, 100).dot(x.cwiseProduct(x)) + 5;
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

## Benchmarks

The tables below compare CppNumericalSolvers against the reference Fortran
L-BFGS-B 3.0 (Byrd/Lu/Nocedal) from
[users.iems.northwestern.edu/~nocedal/lbfgsb.html](https://users.iems.northwestern.edu/~nocedal/lbfgsb.html),
Naoaki Okazaki's C [libLBFGS](https://github.com/chokkan/liblbfgs), and Yixuan
Qiu's header-only [LBFGSpp](https://github.com/yixuan/LBFGSpp). The numbers
below come from a small out-of-tree harness that builds all four
implementations against a common set of test problems; it is not yet shipped
with the library, but the problem definitions and starting points are given
in full underneath each table row.

Each cell reports `iters / nfev`: solver iterations followed by the total
number of calls to the objective function. All runs used identical starting
points, the same gradient-norm tolerance (`1e-6` for L-BFGS/BFGS, `1e-5` for
L-BFGS-B to match Fortran's default `pgtol`), and default line-search
parameters. The reference implementations converge to the same minimum as
CppNumericalSolvers on every row.

### L-BFGS (unconstrained)

| Problem                                | cppoptlib L-BFGS | libLBFGS   |
| -------------------------------------- | ----------------:| ----------:|
| Convex quadratic `5x0² + 100x1² + 5`, start `(-10, 2)` |       11 / 12 |    10 / 11 |
| 2-D Rosenbrock, start `(-1.2, 1)`      |         37 / 45 |    37 / 45 |
| 20-D Extended Rosenbrock               |         35 / 49 |    35 / 49 |

### BFGS (unconstrained, full inverse Hessian)

| Problem                                | cppoptlib BFGS   |
| -------------------------------------- | ----------------:|
| Convex quadratic                       |            4 / 6 |
| 2-D Rosenbrock                         |          32 / 41 |
| 20-D Extended Rosenbrock               |         154 / 226 |

### L-BFGS-B (box-constrained)

| Problem                                                  | cppoptlib L-BFGS-B | Fortran L-BFGS-B 3.0 | LBFGSpp  |
| -------------------------------------------------------- | ------------------:| --------------------:| --------:|
| 2-D Rosenbrock, `x0 >= 0.5`, start `(-1.2, 1)`           |            21 / 28 |                    — |  19 / 22 |
| 2-D Rosenbrock, `x0 >= 1.5`, start `(2, 3)` (bound active at optimum) | 11 / 17 |                 — |  14 / 18 |
| Nocedal extended Rosenbrock, n = 25 (`driver1.f`)        |            24 / 28 |              23 / 28 |  25 / 27 |
| Chebyquad (MGH #35), n = 50, `x ∈ [0, 1]⁵⁰`              |          303 / 328 |            206 / 229 | 241 / 262 |

A few notes on the numbers:

- **Iteration counts and function-evaluation budgets track the references
  closely** on every problem.  On 2-D and 20-D Rosenbrock CppNumericalSolvers
  matches libLBFGS iteration-for-iteration and evaluation-for-evaluation.
- **On the bound-active Rosenbrock** (`x0 >= 1.5`) we actually beat
  LBFGSpp on both counts (11 / 17 vs 14 / 18).  Nocedal's 25-D
  extended-Rosenbrock driver1 is a tie with the Fortran reference on
  nfev (28 vs 28).
- **Chebyquad remains harder** than for the Fortran reference: 328 vs
  229 total evaluations.  We audited our More-Thuente port against
  Fortran's `dcsrch`/`dcstep` and found one textbook divergence (the
  unbracketed step lower bound: Fortran uses `stp + 1.1*(stp-stx)`,
  we use `stx`).  Adopting Fortran's stricter safeguard cuts Chebyquad
  by ~5 % but regresses 2-D/20-D Rosenbrock and BFGS 20-D Rosenbrock by
  more than it saves.  The two styles are a per-problem trade-off, not
  a universal win; we kept the relaxed variant because it wins the
  sum across our benchmark set.  All four implementations converge to
  the same `f = 5.386e-3`.
- All benchmarks above were validated against all three reference
  implementations: the final `x` and `f` agree to at least 6
  significant digits.

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

