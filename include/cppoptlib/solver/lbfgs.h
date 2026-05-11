// CPPNumericalSolvers - A lightweight C++ numerical optimization library
// Copyright (c) 2014    Patrick Wieschollek + Contributors
// Licensed under the MIT License (see below).
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Author: Patrick Wieschollek
//
// More details can be found in the project documentation:
// https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
#define INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_

#include <algorithm>
#include <cmath>
#include <utility>

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {

template <typename FunctionType, int m = 10,
          template <class, int> class LineSearch = linesearch::MoreThuente>
class Lbfgs
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(
      FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::First ||
          FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::Second,
      "L-BFGS only supports first- or second-order differentiable functions");

 public:
  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using ProgressType = typename Superclass::ProgressType;

  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;
  using MatrixType = typename FunctionType::MatrixType;

 private:
  // Storage for the correction pairs using Eigen matrices.
  using memory_MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, m>;
  using memory_VectorType = Eigen::Matrix<ScalarType, 1, m>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Superclass::Superclass;

  void InitializeSolver(const FunctionType& /*function*/,
                        const StateType& initial_state) override {
    const size_t dim = initial_state.x.rows();
    x_diff_memory_ = memory_MatrixType::Zero(dim, m);
    grad_diff_memory_ = memory_MatrixType::Zero(dim, m);
    alpha.resize(m);
    // Scratch buffers used every outer iteration.  Pre-sizing them
    // once here avoids a malloc/free pair per OptimizationStep call.
    x_diff_scratch_.resize(dim);
    grad_diff_scratch_.resize(dim);
    search_direction_scratch_.resize(dim);
    // Reset the circular buffer:
    mem_count_ = 0;
    mem_pos_ = 0;
    scaling_factor_ = 1;
  }

  StateType OptimizationStep(const FunctionType& function,
                             const StateType& current,
                             const ProgressType& /*progress*/) override {
    constexpr ScalarType eps = std::numeric_limits<ScalarType>::epsilon();
    const ScalarType relative_eps =
        static_cast<ScalarType>(eps) *
        std::max<ScalarType>(ScalarType{1.0}, current.x.norm());

    // --- Preconditioner for the two-loop recursion ---
    // If second-order information is available, build a diagonal
    // preconditioner `M^{-1} = diag(|H|)^{-1}` that will be applied to the
    // *center* of the two-loop recursion (Morales & Nocedal 2000,
    // "Automatic preconditioning by limited memory quasi-Newton updating").
    // Crucially, do NOT precondition the starting vector of the recursion:
    // the stored `(s_k, y_k)` pairs live in the unpreconditioned space, so
    // mixing a preconditioned `q_0` with unpreconditioned `s, y` corrupts
    // the recursion.  When no Hessian is available the preconditioner is
    // the identity and the usual Cholesky-style scalar `scaling_factor_` is
    // used as `H_0` instead.
    //
    // Performance: `preconditioner_diagonal` is only consumed inside the
    // `if constexpr` second-order branch.  For first-order problems --
    // the common case on this benchmark's 83-problem suite -- allocating
    // a fresh dynamic-sized vector every outer iteration costs ~5 percent
    // of total run time at small problem sizes.  We hoist the allocation
    // into the second-order branch so first-order paths pay nothing for a
    // preconditioner they do not use.
    constexpr bool has_diagonal_preconditioner =
        FunctionType::Differentiability ==
        cppoptlib::function::DifferentiabilityMode::Second;
    // Read the cached (value, gradient) from the populated FunctionState.
    // For second-order mode we still need the Hessian diagonal, which is
    // not stored on the state, so evaluate once to get it alongside a
    // fresh gradient.  For first-order the gradient is read from the
    // cache and no evaluation happens here -- we take a const reference
    // to avoid the per-iteration vector copy the previous `VectorType
    // current_gradient = current.gradient;` incurred.
    const VectorType* current_gradient_ptr;
    VectorType current_gradient_second_order;
    VectorType preconditioner_diagonal;
    if constexpr (has_diagonal_preconditioner) {
      MatrixType current_hessian;
      function(current.x, &current_gradient_second_order, &current_hessian);
      preconditioner_diagonal =
          current_hessian.diagonal().cwiseAbs().array() + eps;
      preconditioner_diagonal = preconditioner_diagonal.cwiseInverse();
      current_gradient_ptr = &current_gradient_second_order;
    } else {
      current_gradient_ptr = &current.gradient;
    }
    const VectorType& current_gradient = *current_gradient_ptr;

    // --- Two-Loop Recursion ---
    // Start from the raw gradient (unpreconditioned); the Morales-Nocedal
    // preconditioner, if any, is applied below at the center of the loop.
    // Reuse a scratch vector to avoid a per-iteration heap allocation.
    search_direction_scratch_ = current_gradient;
    VectorType& search_direction = search_direction_scratch_;

    // Number of corrections available for the two-loop recursion.  Use every
    // stored pair -- including the newest pair added at the end of the
    // previous step -- because it carries the most recent curvature
    // information and is exactly the pair that Nocedal & Wright Algorithm 7.4
    // consumes first in the backward pass.
    const int k = static_cast<int>(mem_count_);

    // --- First Loop (Backward Pass) ---
    // Iterate over stored corrections in reverse chronological order.
    for (int i = k - 1; i >= 0; i--) {
      // Compute the index in chronological order.
      // When mem_count_ < m, corrections are stored in order [0 ...
      // mem_count_-1]. When full, they are stored cyclically starting at
      // mem_pos_ (oldest) up to (mem_pos_ + m - 1) mod m.
      int idx = (mem_count_ < m ? i : ((mem_pos_ + i) % m));
      const ScalarType denom =
          x_diff_memory_.col(idx).dot(grad_diff_memory_.col(idx));
      if (std::abs(denom) < eps) {
        continue;
      }
      const ScalarType rho = 1.0 / denom;
      alpha(i) = rho * x_diff_memory_.col(idx).dot(search_direction);
      search_direction -= alpha(i) * grad_diff_memory_.col(idx);
    }

    // Apply the initial Hessian approximation.  With a diagonal
    // preconditioner we use `H_0 = M^{-1}` (element-wise scaling).
    // Without one we fall back to the scalar Cholesky estimate
    // `gamma = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}`.
    if (has_diagonal_preconditioner) {
      search_direction =
          preconditioner_diagonal.asDiagonal() * search_direction;
    } else {
      search_direction *= scaling_factor_;
    }

    // --- Second Loop (Forward Pass) ---
    for (int i = 0; i < k; i++) {
      int idx = (mem_count_ < m ? i : ((mem_pos_ + i) % m));
      const ScalarType denom =
          x_diff_memory_.col(idx).dot(grad_diff_memory_.col(idx));
      if (std::abs(denom) < eps) {
        continue;
      }
      const ScalarType rho = 1.0 / denom;
      const ScalarType beta =
          rho * grad_diff_memory_.col(idx).dot(search_direction);
      search_direction += x_diff_memory_.col(idx) * (alpha(i) - beta);
    }

    // Check descent direction validity.
    ScalarType descent_direction = -current_gradient.dot(search_direction);
    // Initial line-search step.  When no curvature information exists yet
    // (iteration zero, or right after a fallback-to-steepest reset), the
    // raw gradient direction has unknown scale so normalize the first trial
    // step as `1 / ||d||`.  Once we have at least one correction pair, the
    // two-loop direction already carries the Hessian scaling, so the
    // standard choice `alpha_init = 1` gives full Newton-like steps that
    // are accepted in one function evaluation for well-conditioned problems.
    ScalarType alpha_init = ScalarType{1};
    if (mem_count_ == 0) {
      const ScalarType search_direction_norm = search_direction.norm();
      alpha_init = (search_direction_norm > eps)
                       ? ScalarType{1} / search_direction_norm
                       : ScalarType{1};
    }
    if (!std::isfinite(descent_direction) ||
        descent_direction > -eps * relative_eps) {
      // Fall back to steepest descent if necessary.
      search_direction = -current_gradient;
      // Reset the correction history if the descent is invalid.
      mem_count_ = 0;
      mem_pos_ = 0;
      const ScalarType gradient_norm = current_gradient.norm();
      alpha_init =
          (gradient_norm > eps) ? ScalarType{1} / gradient_norm : ScalarType{1};
    }

    // Perform a line search.  The incoming `current` already carries
    // `(value, gradient)` at `current.x`, and the `State`-returning
    // overload of `Search` produces a `next` whose `(value, gradient)` are
    // captured from the line search's last internal evaluation.  No
    // redundant evaluations in either direction.
    const StateType next = LineSearch<FunctionType, 1>::Search(
        current, -search_direction, function, alpha_init);

    // Guard: if the line search landed on a non-finite objective
    // (NaN or Inf), the iterate is irrecoverable.  Return the last
    // finite state so the outer stopping criterion fires on the
    // zero x-delta and the caller gets a usable result rather than
    // grinding through thousands of NaN iterations.
    if (!std::isfinite(next.value)) {
      return current;
    }

    // Compute the differences for the new correction pair.  Use
    // scratch buffers stored on the solver so we do not allocate a
    // fresh vector per outer iteration; at small problem sizes
    // (2-50 D) those per-iteration allocations accounted for ~10
    // percent of total run time on the trigonometric-10D benchmark.
    x_diff_scratch_.noalias() = next.x - current.x;
    grad_diff_scratch_.noalias() = next.gradient - current_gradient;
    const VectorType& x_diff = x_diff_scratch_;
    const VectorType& grad_diff = grad_diff_scratch_;

    // --- Curvature Condition Check ---
    // L-BFGS requires `s^T y > 0` for every stored `(s, y)` pair so the
    // implicit inverse-Hessian approximation stays positive definite
    // (Nocedal & Wright 7.4).  The More-Thuente line search enforces this
    // analytically via its curvature condition, but finite-precision noise
    // can still produce a tiny negative or zero `s^T y` near convergence;
    // accept the pair iff `s^T y > eps_machine * ||s|| * ||y||`.  This
    // matches the (unconditional in exact arithmetic) update used by
    // Nocedal's Fortran L-BFGS and libLBFGS.  The earlier cautious update
    // `s^T y > ||s||^2 * ||g|| * 1e-6` was too aggressive on badly-scaled
    // problems with large `||g||` (e.g. MGH 10 Meyer rejected 76% of
    // otherwise-valid pairs, crippling the history buffer).
    const ScalarType sy = x_diff.dot(grad_diff);
    const ScalarType sy_threshold = eps * x_diff.norm() * grad_diff.norm();
    if (sy > sy_threshold) {
      // Add the new correction pair into the circular buffer.
      if (mem_count_ < static_cast<size_t>(m)) {
        // Still have free space.
        x_diff_memory_.col(mem_count_) = x_diff;
        grad_diff_memory_.col(mem_count_) = grad_diff;
        mem_count_++;
      } else {
        // Buffer full; overwrite the oldest correction.
        x_diff_memory_.col(mem_pos_) = x_diff;
        grad_diff_memory_.col(mem_pos_) = grad_diff;
        mem_pos_ = (mem_pos_ + 1) % m;
      }
    }
    // Update the scaling factor (initial Hessian approximation `H_0`).  The
    // standard L-BFGS choice is `gamma_k = s_k^T y_k / y_k^T y_k` (Nocedal &
    // Wright 7.20).  If the line search failed and produced a zero step --
    // in which case `x_diff = grad_diff = 0` -- we have no information to
    // update `H_0`, so keep the previous scaling rather than clobber it with
    // a huge fallback (which would cause the next iteration to take a
    // catastrophic step).  We also keep the previous scaling when the new
    // estimate is non-finite or wildly large.
    constexpr ScalarType fallback_value = ScalarType(1e7);
    const ScalarType grad_diff_norm_sq = grad_diff.dot(grad_diff);
    if (grad_diff_norm_sq > eps) {
      const ScalarType temp_scaling = grad_diff.dot(x_diff) / grad_diff_norm_sq;
      if (std::isfinite(temp_scaling) &&
          std::abs(temp_scaling) <= fallback_value) {
        scaling_factor_ = std::max(temp_scaling, eps);
      }
      // else: keep previous scaling_factor_.
    }
    // else: grad_diff is effectively zero -- no new curvature info, keep
    // previous scaling_factor_.

    return next;
  }

 private:
  memory_MatrixType x_diff_memory_;
  memory_MatrixType grad_diff_memory_;
  // Circular buffer state:
  size_t mem_count_ = 0;  // Number of corrections stored so far (max m).
  size_t mem_pos_ = 0;    // Index of the oldest correction in the buffer.

  memory_VectorType
      alpha;  // Storage for the coefficients in the two-loop recursion.
  ScalarType scaling_factor_ = 1;

  // Scratch vectors reused across every `OptimizationStep` call.
  // Allocated once in `InitializeSolver`, resized to the problem
  // dimension; their contents are fully overwritten on each entry
  // so no cross-iteration state is stored here.  At small problem
  // sizes these save a `malloc + free` per outer iteration.
  VectorType x_diff_scratch_;
  VectorType grad_diff_scratch_;
  VectorType search_direction_scratch_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_LBFGS_H_
