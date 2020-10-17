CppOptimizationLibrary (A header-only C++17 optimization library)

[![Build Status](https://ci.patwie.com/api/badges/PatWie/CppNumericalSolvers/status.svg?ref=refs/heads/v2)](https://ci.patwie.com/PatWie/CppNumericalSolvers)

It has been now 6 years since the initial release. I did some mistakes in the previous design of this library and some features felt a bit ad-hoc. Given that C++14 is around and C++17 will become mainstream, I will take the opportunity to correct some of these mistakes and simplify things here in this v2-branch.

This branch is under development.

> For the previous fully-tested version please refer to the [master-branch](https://github.com/PatWie/CppNumericalSolvers/tree/master) of this repository.


Have you ever looked for a C++ function *fminsearch*, which is easy to use without adding tons of dependencies and without editing many setting-structs and without dependencies?

Want a full example?

```cpp
    using FunctionXd = cppoptlib::function::Function<double>;

    class Rosenbrock : public FunctionXd {
      public:

        double operator()(const Eigen::VectorXd &x) {
            const double t1 = (1 - x[0]);
            const double t2 = (x[1] - x[0] * x[0]);
            return   t1 * t1 + 100 * t2 * t2;
        }
    };
    int main(int argc, char const *argv[]) {
        using Solver = cppoptlib::solver::Bfgs<Rosenbrock>;

        Rosenbrock f;
        Eigen::VectorXd x(2);
        x << -1, 2;

        // Evaluate
        auto state = f.Eval(x);
        std::cout << f(x) << " = " << state.value << std::endl;
        std::cout << state.x << std::endl;
        std::cout << state.gradient << std::endl;
        std::cout << state.hessian << std::endl;


        std::cout << cppoptlib::utils::IsGradientCorrect(f, x) << std::endl;
        std::cout << cppoptlib::utils::IsHessianCorrect(f, x) << std::endl;

        Solver solver;

        auto[solution, solver_state] = solver.Minimize(f, x);
        std::cout << "argmin " << solution.x.transpose() << std::endl;
        std::cout << "f in argmin " << solution.value << std::endl;
        std::cout << "iterations " << solver_state.num_iterations << std::endl;
        std::cout << "solver status " << solver_state.status << std::endl;
        return 0;
    }
```

To use another solver, simply replace `BfgsSolver` by another name.

# Changes

- Instead of nesting information in each class, this implementation will handle and update states of functions and solvers.
- This will follow clang-format google-code-style and will be compliant cpplint.
- This will drop Support for TensorFlow and Matlab (maybe Python will be an option).

[eigen3]: http://eigen.tuxfamily.org/
[bazel]: https://bazel.build/
[matlab]: http://www.mathworks.de/products/matlab/
[tensorflow]: https://www.tensorflow.org/

# References

**L-BFGS-B**: A LIMITED MEMORY ALGORITHM FOR BOUND CONSTRAINED OPTIMIZATION
*Richard H. Byrd, Peihuang Lu, Jorge Nocedal and Ciyou Zhu*

**L-BFGS**: Numerical Optimization, 2nd ed. New York: Springer
*J. Nocedal and S. J. Wright*

# Citing this implementation

I see some interests in citing this implementation. Please use the following bibtex entry, if you consider to cite this implementation:

```bibtex
@misc{wieschollek2016cppoptimizationlibrary,
  title={CppOptimizationLibrary},
  author={Wieschollek, Patrick},
  year={2016},
  howpublished={\url{https://github.com/PatWie/CppNumericalSolvers}},
}
```
