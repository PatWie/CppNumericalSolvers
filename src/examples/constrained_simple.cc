// Copyright 2025, https://github.com/PatWie/CppNumericalSolvers
#include <iostream>

#include "include/cppoptlib/constrained_function.h"
#include "include/cppoptlib/function.h"
#include "include/cppoptlib/solver/augmented_lagrangian.h"
#include "include/cppoptlib/solver/lbfgs.h"

using Function2d = cppoptlib::function::Function<
    double, 2, cppoptlib::function::Differentiability::First>;

class SumObjective : public Function2d {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  state_t operator()(const vector_t &x) const override {
    state_t state;
    state.x = x;
    state.value = x.sum();
    state.gradient = vector_t::Ones(2);
    return state;
  }
};

class InsideCircleConstraint : public Function2d {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Enforces x[0]^2+x[1]^2 <= 2 (inside the circle)
  state_t operator()(const vector_t &x) const override {
    state_t state;
    state.x = x;
    state.value = 2 - x.squaredNorm();
    state.gradient = -2 * x;
    return state;
  }
};

class CircleBoundaryConstraint : public Function2d {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Enforces x[0]^2+x[1]^2 == 2 (on the circle boundary)
  state_t operator()(const vector_t &x) const override {
    state_t state;
    state.x = x;
    state.value = x.squaredNorm() - 2;
    state.gradient = 2 * x;
    return state;
  }
};

int main() {
  using InnerSolver = cppoptlib::solver::Lbfgs<Function2d>;

  SumObjective::vector_t x;
  x << 2, 10;

  // Problem:
  //  min   f(x) = x[0] + x[1]        // SumObjective
  //  s.t.      x[0]^2+x[1]^2 <= 2    // InsideCircleConstraint
  //            x[0]^2+x[1]^2 == 2    // CircleBoundaryConstraint
  SumObjective f;
  cppoptlib::function::NonNegativeConstraint<InsideCircleConstraint> c1;
  cppoptlib::function::ZeroConstraint<CircleBoundaryConstraint> c2;

  using Function2dC =
      cppoptlib::function::ConstrainedFunction<Function2d,
                                               /* num constraints */ 2>;

  Function2dC L(&f, {&c1, &c2});
  // cppoptlib::function::ConstrainedState<Function2dC>
  const auto state = L(x, {0.0, 0.0}, 10.);

  cppoptlib::solver::AugmentedLagrangian<Function2dC, InnerSolver> solver;

  auto [solution, solver_state] = solver.Minimize(L, state);
  std::cout << "f in argmin " << solution.value << std::endl;
  // Supposed to be [-1, -1].
  std::cout << "argmin " << solution.x.transpose() << std::endl;
  std::cout << "iterations " << solver_state.num_iterations << std::endl;
  std::cout << "status " << solver_state.status << std::endl;

  return 0;
}
