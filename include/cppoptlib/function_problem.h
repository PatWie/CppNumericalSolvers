#ifndef INCLUDE_CPPOPTLIB_FUNCTION_PROBLEM_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_PROBLEM_H_

#include "function_problem.h"
#include <Eigen/Core>
#include <type_traits>
#include <vector>

namespace cppoptlib::function {

// template <typename TScalar, DifferentiabilityMode ModeObj,
//           DifferentiabilityMode ModeConstr>
// struct UnconstrainedOptimizationProblem {
//   static constexpr DifferentiabilityMode Differentiability = ModeObj;
//
//   const AnyFunction<TScalar, ModeObj> objective; // f(x)
//
//   UnconstrainedOptimizationProblem(const AnyFunction<TScalar, ModeObj> obj)
//       : objective(std::move(obj)) {}
// };

// ConstrainedOptimizationProblem represents a constrained optimization problem:
//   minimize f(x)
//   subject to: c(x) == 0  and  c(x) >= 0
//
template <typename TScalar, DifferentiabilityMode ModeObj,
          DifferentiabilityMode ModeConstr, int TDimension>
struct ConstrainedOptimizationProblem {

  static constexpr int Dimension = TDimension;

  using ScalarType = TScalar;
  using VectorType = Eigen::Matrix<ScalarType, Dimension, 1>;
  using MatrixType = Eigen::Matrix<ScalarType, Dimension, Dimension>;

  using ObjectiveFunctionType = AnyFunction<TScalar, ModeObj, TDimension>;
  using ConstraintFunctionType = AnyFunction<TScalar, ModeConstr, TDimension>;
  static constexpr DifferentiabilityMode Differentiability =
      MinDifferentiabilityMode<ModeObj, ModeConstr>::value;

  const AnyFunction<TScalar, ModeObj, TDimension> objective; // f(x)
  const std::vector<AnyFunction<TScalar, ModeConstr, TDimension>>
      equality_constraints; // c(x) == 0
  const std::vector<AnyFunction<TScalar, ModeConstr, TDimension>>
      inequality_constraints; // c(x) >= 0

  ConstrainedOptimizationProblem(
      const AnyFunction<TScalar, ModeObj, TDimension> obj,
      const std::vector<AnyFunction<TScalar, ModeConstr, TDimension>>
          eq_constraints = {},
      const std::vector<AnyFunction<TScalar, ModeConstr, TDimension>>
          ineq_constraints = {})
      : objective(std::move(obj)),
        equality_constraints(std::move(eq_constraints)),
        inequality_constraints(std::move(ineq_constraints)) {}
};

// A simple deduction guide example, assuming F and G have a static
// Differentiability member.
// Deduction guide for two-argument constructor using an initializer list for
// equality constraints.
// template <typename TScalar, DifferentiabilityMode ModeObj,
//           DifferentiabilityMode ModeConstr, int TDim>
// ConstrainedOptimizationProblem(
//     const AnyFunction<TScalar, ModeObj, TDim> &,
//     std::initializer_list<AnyFunction<TScalar, ModeConstr, TDim>>)
//     -> ConstrainedOptimizationProblem<TScalar, ModeObj, ModeConstr, TDim>;

// Deduction guide for three-argument constructor using initializer lists.
template <typename TScalar, DifferentiabilityMode ModeObj,
          DifferentiabilityMode ModeConstr, int TDim>
ConstrainedOptimizationProblem(
    const AnyFunction<TScalar, ModeObj, TDim> &,
    std::initializer_list<AnyFunction<TScalar, ModeConstr, TDim>>,
    std::initializer_list<AnyFunction<TScalar, ModeConstr, TDim>>)
    -> ConstrainedOptimizationProblem<TScalar, ModeObj, ModeConstr, TDim>;

} // namespace cppoptlib::function

#endif //  INCLUDE_CPPOPTLIB_FUNCTION_PROBLEM_H_
