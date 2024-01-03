# CppNumericalSolvers (A header-only C++17 optimization library)

CppNumericalSolvers stands as a robust and efficient header-only C++17
optimization library, meticulously crafted to address numerical optimization
challenges. This library offers a suite of powerful solvers for optimization
problems, placing a strong emphasis on simplicity, adherence to modern C++
standards, and seamless integration into projects.

### Example Usage: Minimizing the Rosenbrock Function

Let's delve into a straightforward example that illustrates the ease of
utilizing CppNumericalSolvers for numerical optimization. In this instance, we
will showcase the minimization of the classic Rosenbrock function using the
BFGS solver.

```cpp
using FunctionXd = cppoptlib::function::Function<double>;

/**
 * @brief Definition of the Rosenbrock function for optimization.
 *
 * This class represents the Rosenbrock function, a classic optimization problem.
 */
class Rosenbrock : public FunctionXd {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Computes the value of the Rosenbrock function at a given point.
     *
     * @param x The input vector.
     * @return The value of the Rosenbrock function at the given point.
     */
    double operator()(const Eigen::VectorXd &x) const {
        const double t1 = (1 - x[0]);
        const double t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    }

    // Gradient and Hessian implementation can be omitted.
};

int main(int argc, char const *argv[]) {
    // Create an instance of the Rosenbrock function.
    Rosenbrock f;

    // Initial guess for the solution.
    Eigen::VectorXd x(2);
    x << -1, 2;
    std::cout << "Initial point: " << x << std::endl;
    std::cout << "Function value at initial point: " << f(x) << std::endl;

    // Check the correctness of the gradient and Hessian (against numerical implementation).
    std::cout << "Is Gradient correctly implemented? " << cppoptlib::utils::IsGradientCorrect(f, x) << std::endl;
    std::cout << "Is Hessian correctly implemented? " << cppoptlib::utils::IsHessianCorrect(f, x) << std::endl;

    // Evaluate
    auto state = f.Eval(x);
    std::cout << "Gradient at initial point: " << state.gradient << std::endl;
    if (state.hessian) {
        std::cout << "Hessian at initial point: " << *(state.hessian) << std::endl;
    }

    // Minimize the Rosenbrock function using the BFGS solver.
    using Solver = cppoptlib::solver::Bfgs<Rosenbrock>;
    Solver solver;
    auto [solution, solver_state] = solver.Minimize(f, x);

    // Display the results of the optimization.
    std::cout << "Optimal solution: " << solution.x.transpose() << std::endl;
    std::cout << "Optimal function value: " << solution.value << std::endl;
    std::cout << "Number of iterations: " << solver_state.num_iterations << std::endl;
    std::cout << "Solver status: " << solver_state.status << std::endl;

    return 0;
}
```

This example demonstrates the usage of the BFGS solver to minimize the
Rosenbrock function. You can easily adapt this code for your specific
optimization problem by defining your objective function and selecting an
appropriate solver from CppNumericalSolvers. Dive into the implementation for
more details on available solvers and advanced usage.

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
