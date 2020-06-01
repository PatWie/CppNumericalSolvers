CppOptimizationLibrary
=================================================================

This branch is under development.

[![Build Status](http://ci.patwie.com/api/badges/PatWie/CppNumericalSolvers/status.svg)](http://ci.patwie.com/PatWie/CppNumericalSolvers)

A *header-only* library with bindings to **C++**.

Have you ever looked for a C++ function *fminsearch*, which is easy to use without adding tons of dependencies and without editing many setting-structs and without dependencies?

Want a full example?

```cpp
    using Problem = Function<double, /*Order*/ 2, /*Order*/ 2>;
    class Rosenbrock : public Problem {
      public:

        using vector_t = typename Problem::vector_t;
        using scalar_t = typename Problem::scalar_t;

        scalar_t operator()(const vector_t &x) {
            const scalar_t t1 = (1 - x[0]);
            const scalar_t t2 = (x[1] - x[0] * x[0]);
            return   t1 * t1 + 100 * t2 * t2;
        }
    };
    int main(int argc, char const *argv[]) {

        using function_t = Rosenbrock;
        using solver_t = BfgsSolver<Rosenbrock>

        function_t f;
        function_t::scalar_t x(2); x << -1, 2;

        solver_t solver;

        function_t::state_t solution;
        solver_t::state_t solver_state;

        std::tie(solution, solver_state) = solver.minimize(f, x);
        std::cout << "argmin      " << solution.x.transpose() << std::endl;
        std::cout << "f in argmin " << f(solution.x) << std::endl;
        std::cout << "took iterations " << solver_state.num_iterations << std::endl;
        return 0;
    }
```

To use another solver, simply replace `BfgsSolver` by another name.

# Changes

- Instead of nesting information in each class, this implementation will handle and update states of functions and solvers.
- Drop Support for TensorFlow and Matlab (maybe consider Python Support)

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
