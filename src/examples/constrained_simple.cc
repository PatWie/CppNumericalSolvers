// Copyright 2025, https://github.com/PatWie/CppNumericalSolvers
#include <iostream>

#include "include/cppoptlib/constrained_function.h"
#include "include/cppoptlib/function.h"
#include "include/cppoptlib/solver/augmented_lagrangian.h"
#include "include/cppoptlib/solver/lbfgs.h"

template <class T>
using Function2d = cppoptlib::function::Function<
    double, 2, T, cppoptlib::function::Differentiability::First>;

class SumObjective : public Function2d<SumObjective> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  scalar_t operator()(const vector_t &x, vector_t *gradient = nullptr) const {
    if (gradient) {
      *gradient = vector_t::Ones(2);
    }
    return x.sum();
  }
};

class InsideCircleConstraint : public Function2d<InsideCircleConstraint> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Enforces x[0]^2+x[1]^2 <= 2 (inside the circle)
  scalar_t operator()(const vector_t &x, vector_t *gradient = nullptr) const {
    if (gradient) {
      *gradient = -2 * x;
    }
    return 2 - x.squaredNorm();
  }
};

class CircleBoundaryConstraint : public Function2d<CircleBoundaryConstraint> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Enforces x[0]^2+x[1]^2 == 2 (on the circle boundary)
  scalar_t operator()(const vector_t &x, vector_t *gradient = nullptr) const {
    if (gradient) {
      *gradient = 2 * x;
    }
    return x.squaredNorm() - 2;
  }
};

int main() {
  SumObjective::vector_t x;
  x << 2, 10;

  // Problem:
  //  min   f(x) = x[0] + x[1]        // SumObjective
  //  s.t.      x[0]^2+x[1]^2 <= 2    // InsideCircleConstraint
  //            x[0]^2+x[1]^2 == 2    // CircleBoundaryConstraint
  SumObjective f;
  cppoptlib::function::NonNegativeConstraint<InsideCircleConstraint> c1;
  cppoptlib::function::ZeroConstraint<CircleBoundaryConstraint> c2;

  const auto L = cppoptlib::function::BuildConstrainedProblem(&f, &c1, &c2);

  // cppoptlib::solver::Lbfgs<decltype(f)> inner_solver;
  // cppoptlib::solver::AugmentedLagrangian<decltype(L), decltype(inner_solver)>
  //     solver(inner_solver);
  //
  // const auto initial_state = L.GetState(x, {0.0, 0.0}, 10.);
  // auto [solution, solver_state] = solver.Minimize(L, initial_state);
  // std::cout << "f in argmin " << f(solution.x) << std::endl;
  // // Supposed to be [-1, -1].
  // std::cout << "argmin " << solution.x.transpose() << std::endl;
  // std::cout << "iterations " << solver_state.num_iterations << std::endl;
  // std::cout << "status " << solver_state.status << std::endl;

  return 0;
}
