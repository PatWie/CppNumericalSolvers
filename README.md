# CppNumericalSolvers (A header-only C++17 optimization library)

CppNumericalSolvers is a header-only C++17 optimization library providing a
suite of solvers for both unconstrained and constrained optimization problems.
The library is designed for efficiency, modern C++ compliance, and easy
integration into existing projects. It is distributed under a permissive
license, making it suitable for commercial use.

### Example Usage: Minimizing the Rosenbrock Function

Let minimize the classic Rosenbrock function using the BFGS solver.

```cpp
/**
 * @brief Alias for a 2D function supporting second-order differentiability in cppoptlib.
 *
 * This defines a function template specialized for a 2-dimensional input vector,
 * which supports evaluation of the function value and its derivatives.
 *
 * The differentiability level is determined by the Differentiability enum:
 *  - Differentiability::First: Supports first-order derivatives (i.e., the gradient)
 *    without computing the Hessian. This is useful for optimization methods that do not
 *    require second-order information, saving computational effort.
 *  - Differentiability::Second: Supports second-order derivatives, meaning that both the
 *    gradient and Hessian are computed. This level is needed for methods that rely on curvature
 *    information.
 *
 * In this alias, Differentiability::Second is used, so both the gradient and Hessian are
 * assumed to be implemented.
 */
using Function2D = cppoptlib::function::Function<double, 2, cppoptlib::function::Differentiability::Second>;

/**
 * @brief Implementation of a quadratic function with optional gradient and Hessian computation.
 *
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
 *
 * This implementation computes the gradient and Hessian only if the corresponding pointers
 * are provided by the caller.
 */
class Function : public Function2D {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Evaluates the function and optionally its gradient and Hessian.
   *
   * @param x Input vector.
   * @param gradient (Optional) Pointer to store the computed gradient.
   * @param hessian  (Optional) Pointer to store the computed Hessian.
   * @return The function value f(x).
   */
  scalar_t operator()(const vector_t &x, vector_t *gradient = nullptr,
                        matrix_t *hessian = nullptr) const override {

    // Even for functions declared as Differentiability::First, the gradient is not always required.
    // To save computation, we only calculate and store the gradient if a non-null pointer is provided.
    if (gradient) {
      gradient->resize(x.size());
      // The gradient components:
      // ∂f/∂x₀ = 2 * 5 * x₀ = 10 * x₀
      // ∂f/∂x₁ = 2 * 100 * x₁ = 200 * x₁
      (*gradient)[0] = 2 * 5 * x[0];
      (*gradient)[1] = 2 * 100 * x[1];
    }

    // If the Hessian is requested, compute and store it.
    // (This is applicable only for Differentiability::Second)
    if (hessian) {
      hessian->resize(x.size(), x.size());
      // Set the Hessian components:
      // ∂²f/∂x₀² = 10, ∂²f/∂x₁² = 200, and the off-diagonals are 0.
      (*hessian)(0, 0) = 10;
      (*hessian)(0, 1) = 0;
      (*hessian)(1, 0) = 0;
      (*hessian)(1, 1) = 200;
    }

    // Return the function value: f(x) = 5*x₀² + 100*x₁² + 5.
    return 5 * x[0] * x[0] + 100 * x[1] * x[1] + 5;
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
    auto state = f.GetState(x);
    std::cout << "Function value at initial point: " << f(x) << std::endl;
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
