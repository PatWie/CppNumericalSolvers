#include <cstdlib>
#include <vector>

#include "Eigen/Core"
#include "fmt/core.h"
#include "function.h"
#include "solver/bfgs.h"
#include "solver/conjugated_gradient_descent.h"
#include "solver/gradient_descent.h"
#include "solver/lbfgs.h"
#include "solver/lbfgsb.h"
#include "solver/newton_descent.h"
#include "solver/solver.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

namespace {

using FunctionXd = cppoptlib::function::Function<double>;

// https://en.wikipedia.org/wiki/Rosenbrock_function
struct Rosenbrock : public FunctionXd {
  using FunctionXd::hessian_t;
  using FunctionXd::vector_t;

  scalar_t operator()(const vector_t& x) const override {
    const double t1 = (a - x[0]);
    const double t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + b * t2 * t2;
  }
  static constexpr double a = 1;
  static constexpr double b = 100;
};

template <typename Solver>
auto solve(const std::vector<double>& values) {
  Rosenbrock f;
  Rosenbrock::vector_t x(2);
  x << values[0], values[1];
  auto state = f.Eval(x);
  Solver solver;
  auto step_callback =
      cppoptlib::solver::GetEmptyStepCallback<typename Rosenbrock::scalar_t,
                                              typename Rosenbrock::vector_t,
                                              typename Rosenbrock::hessian_t>();
  solver.SetStepCallback(step_callback);
  return solver.Minimize(f, x);
}
}  // namespace

TEST_CASE("NewtonDescent") {
  using Solver = cppoptlib::solver::NewtonDescent<Rosenbrock>;

  SUBCASE("Cannot converge") {
    const std::vector<double> x{-100, 4};
    const auto& [sol, state] = solve<Solver>(x);
    CHECK_EQ(cppoptlib::solver::Status::XDeltaViolation, state.status);
  }

  SUBCASE("Converge") {
    const std::vector<double> x{2, 3};
    const auto& [sol, state] = solve<Solver>(x);
    CHECK_EQ(cppoptlib::solver::Status::GradientNormViolation, state.status);
    Eigen::VectorXd results(2);
    results << 1, 1;
    constexpr double epsilon = 1e-6;
    auto relerr = (results - sol.x).norm() / results.norm();
    CHECK(relerr == doctest::Approx(0.0).epsilon(epsilon));
  }
}

TEST_CASE("GradientDescent") {
  using Solver = cppoptlib::solver::GradientDescent<Rosenbrock>;

  SUBCASE("Simple usage") {
    const std::vector<double> x{100, -4};
    const auto& [sol, state] = solve<Solver>(x);
    CHECK_EQ(cppoptlib::solver::Status::GradientNormViolation, state.status);
    Eigen::VectorXd results(2);
    results << 1, 1;
    constexpr double epsilon = 1e-3;
    auto relerr = (results - sol.x).norm() / results.norm();
    CHECK(relerr == doctest::Approx(0.0).epsilon(epsilon));
  }
}

TEST_CASE("ConjugateGradientDescent") {
  using Solver = cppoptlib::solver::ConjugatedGradientDescent<Rosenbrock>;

  SUBCASE("Simple usage") {
    const std::vector<double> x{100, -4};
    const auto& [sol, state] = solve<Solver>(x);
    CHECK_EQ(cppoptlib::solver::Status::GradientNormViolation, state.status);
    Eigen::VectorXd results(2);
    results << 1, 1;
    constexpr double epsilon = 1e-4;
    auto relerr = (results - sol.x).norm() / results.norm();
    CHECK(relerr == doctest::Approx(0.0).epsilon(epsilon));
  }
}

TEST_CASE("BFGS") {
  using Solver = cppoptlib::solver::Bfgs<Rosenbrock>;

  SUBCASE("Simple usage") {
    const std::vector<double> x{100, -4};
    const auto& [sol, state] = solve<Solver>(x);
    CHECK_EQ(cppoptlib::solver::Status::GradientNormViolation, state.status);
    Eigen::VectorXd results(2);
    results << 1, 1;
    constexpr double epsilon = 1e-4;
    auto relerr = (results - sol.x).norm() / results.norm();
    CHECK(relerr == doctest::Approx(0.0).epsilon(epsilon));
  }
}

TEST_CASE("L-BFGS") {
  using Solver = cppoptlib::solver::Lbfgs<Rosenbrock>;

  SUBCASE("Simple usage") {
    const std::vector<double> x{100, -4};
    const auto& [sol, state] = solve<Solver>(x);
    CHECK_EQ(cppoptlib::solver::Status::GradientNormViolation, state.status);
    Eigen::VectorXd results(2);
    results << 1, 1;
    constexpr double epsilon = 1e-4;
    auto relerr = (results - sol.x).norm() / results.norm();
    CHECK(relerr == doctest::Approx(0.0).epsilon(epsilon));
  }
}

TEST_CASE("L-BFGS-B") {
  using Solver = cppoptlib::solver::Lbfgs<Rosenbrock>;

  SUBCASE("Simple usage") {
    const std::vector<double> x{100, -4};
    const auto& [sol, state] = solve<Solver>(x);
    CHECK_EQ(cppoptlib::solver::Status::GradientNormViolation, state.status);
    Eigen::VectorXd results(2);
    results << 1, 1;
    constexpr double epsilon = 1e-4;
    auto relerr = (results - sol.x).norm() / results.norm();
    CHECK(relerr == doctest::Approx(0.0).epsilon(epsilon));
  }
}
