# CppNumericalSolvers (A header-only C++17 optimization library)

CppNumericalSolvers is a header-only C++17 optimization library providing a
suite of solvers for both unconstrained and constrained optimization problems.
The library is designed for efficiency, modern C++ compliance, and easy
integration into existing projects. It is distributed under a permissive
license, making it suitable for commercial use.

### Example Usage: Minimizing the Rosenbrock Function

Let minimizate of the classic Rosenbrock function using the BFGS solver.

```cpp
/**
 * @brief Alias for a 2D function with first-order differentiability in cppoptlib.
 *
 * This defines a function template that supports differentiation, specialized for
 * a 2-dimensional input vector.
 */
using Functiond2_dx = cppoptlib::function::Function<
    double, 2, cppoptlib::function::Differentiability::First>;

/**
 * @brief Implementation of the Rosenbrock function with gradient computation.
 *
 * This class represents the Rosenbrock function:
 *
 *     f(x) = (1 - x₁)² + 100 * (x₂ - x₁²)²
 *
 * It includes both function evaluation and its gradient for optimization algorithms.
 * The function has a global minimum at (x₁, x₂) = (1, 1), where f(x) = 0.
 *
 * @tparam T The scalar type (e.g., double or float).
 */
class RosenbrockGradient : public Functiond2_dx {
public:
  /// Eigen macro for proper memory alignment.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Import necessary typedefs from the base class.
  using typename FunctionX2_dx::state_t;
  using typename FunctionX2_dx::scalar_t;
  using typename FunctionX2_dx::vector_t;

  /**
   * @brief Computes the Rosenbrock function value and its gradient at a given point.
   *
   * @param x A 2D Eigen vector representing the input point.
   * @return A state_t object containing the function value, input vector, and gradient.
   */
  state_t operator()(const vector_t &x) const override {
    state_t state;

    // Compute function value: f(x) = (1 - x₁)² + 100 * (x₂ - x₁²)²
    const scalar_t t1 = (1 - x[0]);             // First term: (1 - x₁)
    const scalar_t t2 = (x[1] - x[0] * x[0]);   // Second term: (x₂ - x₁²)

    state.value = t1 * t1 + 100 * t2 * t2;
    state.x = x;  // Store the input vector for reference.

    // Initialize gradient vector (∇f)
    state.gradient = vector_t::Zero(2);

    // Compute partial derivatives:
    // ∂f/∂x₁ = -2(1 - x₁) + 200(x₂ - x₁²)(-2x₁)
    state.gradient[0] = -2 * t1 + 200 * t2 * (-2 * x[0]);

    // ∂f/∂x₂ = 200(x₂ - x₁²)
    state.gradient[1] = 200 * t2;

    return state;
  }
};

int main(int argc, char const *argv[]) {
    // Create an instance of the Rosenbrock function.
    Rosenbrock f;

    // Initial guess for the solution.
    Eigen::VectorXd x(2);
    x << -1, 2;
    std::cout << "Initial point: " << x.transpose() << std::endl;

    // Evaluate
    auto state = f(x);
    std::cout << "Function value at initial point: " << state.value << std::endl;
    std::cout << "Gradient at initial point: " << state.gradient << std::endl;

    // Minimize the Rosenbrock function using the BFGS solver.
    using Solver = cppoptlib::solver::Bfgs<Rosenbrock>;
    Solver solver;
    auto [solution, solver_progress] = solver.Minimize(f, x);

    // Display the results of the optimization.
    std::cout << "Optimal solution: " << solution.x.transpose() << std::endl;
    std::cout << "Optimal function value: " << solution.value << std::endl;
    std::cout << "Number of iterations: " << solver_progress.num_iterations << std::endl;
    std::cout << "Solver status: " << solver_progress.status << std::endl;

    return 0;
}
```

You can easily adapt this code for your specific optimization problem by
defining your objective function and selecting an appropriate solver from
CppNumericalSolvers. Dive into the implementation for more details on available
solvers and advanced usage.


See the examples for a constrained problem.

### References

**L-BFGS-B**: A LIMITED MEMORY ALGORITHM FOR BOUND CONSTRAINED OPTIMIZATION
_Richard H. Byrd, Peihuang Lu, Jorge Nocedal and Ciyou Zhu_

**L-BFGS**: Numerical Optimization, 2nd ed. New York: Springer
_J. Nocedal and S. J. Wright_

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
