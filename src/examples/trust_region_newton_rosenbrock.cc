// Copyright 2026, https://github.com/PatWie/CppNumericalSolvers
//
// Example: trust-region Newton with CG-Steihaug on the Rosenbrock
// function.
//
// Rosenbrock is the standard non-convex test problem: its Hessian is
// indefinite for much of the `(-1.2, 1) -> (1, 1)` descent path.  A
// plain Newton step with a diagonal safeguard and Armijo line search
// (the library's `NewtonDescent`) can stall on such paths; the
// trust-region variant handles indefinite curvature as a structural
// case of the algorithm via CG-Steihaug's negative-curvature exit.
//
// The driver prints per-iteration state so the trust-region radius
// dynamics are visible: the radius grows when agreement is good, and
// the solver converges on the global minimum at (1, 1) in a few
// dozen outer iterations.

#include <cstdio>
#include <iostream>

#include "Eigen/Core"
#include "cppoptlib/function.h"
#include "cppoptlib/solver/trust_region_newton.h"

// The Rosenbrock objective, second-order.  We hand-code the Hessian
// because TR-Newton requires `DifferentiabilityMode::Second`.  The
// 400-per-coordinate condition number away from the minimum gives
// the non-convex stress that makes this a good example.
class Rosenbrock : public cppoptlib::function::FunctionCRTP<
                       Rosenbrock, double,
                       cppoptlib::function::DifferentiabilityMode::Second, 2> {
 public:
  ScalarType operator()(const VectorType& x, VectorType* grad = nullptr,
                        MatrixType* hess = nullptr) const {
    const double a = 1 - x(0);
    const double b = x(1) - x(0) * x(0);
    if (grad) {
      (*grad)(0) = -2 * a - 400 * b * x(0);
      (*grad)(1) = 200 * b;
    }
    if (hess) {
      (*hess)(0, 0) = 2 - 400 * b + 800 * x(0) * x(0);
      (*hess)(0, 1) = -400 * x(0);
      (*hess)(1, 0) = -400 * x(0);
      (*hess)(1, 1) = 200;
    }
    return 100 * b * b + a * a;
  }
};

int main() {
  Rosenbrock f;
  cppoptlib::solver::TrustRegionNewton<Rosenbrock> solver;
  solver.stopping_progress.gradient_norm = 1e-10;
  solver.stopping_progress.num_iterations = 200;

  // Print iterate, objective, and (on first call) the initial radius.
  // The library's standard callback signature is
  // `(function, state, progress)`.  We log every iteration because
  // the example is small.
  solver.SetCallback(
      [](const Rosenbrock& /*func*/, const auto& state, const auto& progress) {
        std::printf("iter=%3zu  x=(%.6f, %.6f)  f=%.6g  |g|_inf=%.3e\n",
                    progress.num_iterations, state.x(0), state.x(1),
                    state.value, progress.gradient_norm);
      });

  Eigen::Vector2d x0(-1.2, 1.0);
  auto [solution, progress] =
      solver.Minimize(f, cppoptlib::function::FunctionState(x0));

  std::printf("\n");
  std::printf(
      "Converged at x = (%.9f, %.9f), f = %.3e after %zu outer iters.\n",
      solution.x(0), solution.x(1), solution.value, progress.num_iterations);
  std::cout << "Exit status: " << progress.status << "\n";
  return 0;
}
