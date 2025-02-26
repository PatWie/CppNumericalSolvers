// Copyright 2025, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_NELDER_MEAD_H_
#define INCLUDE_CPPOPTLIB_SOLVER_NELDER_MEAD_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib::solver {

template <typename function_t>
class NelderMead : public Solver<function_t> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass = Solver<function_t>;
  using state_t = typename function_t::state_t;
  using scalar_t = typename function_t::scalar_t;
  using vector_t = typename function_t::vector_t;
  using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using progress_t = typename Superclass::progress_t;

  // Coefficients for the Nelder–Mead algorithm.
  const scalar_t rho_ = 1.0;    // Reflection coefficient (> 0)
  const scalar_t xi_ = 1.5;     // Expansion coefficient (reduced from 2.0)
  const scalar_t gamma_ = 0.5;  // Contraction coefficient (0 < gamma < 1)
  const scalar_t sigma_ = 0.5;  // Shrink coefficient (0 < sigma < 1)

  // Tolerance for detecting a degenerate simplex.
  const scalar_t degenerate_tol_ = 1e-8;

  using Superclass::Superclass;

  // Initialize the solver with the starting point.
  void InitializeSolver(const state_t &initial_state) override {
    simplex_ = makeInitialSimplex(initial_state.x);
  }

  // Performs one iteration (step) of the Nelder–Mead algorithm.
  // This implementation updates the internal simplex and returns the current
  // best vertex.
  state_t OptimizationStep(const function_t &function, const state_t &current,
                           const progress_t &progress) override {
    const size_t DIM = current.x.rows();
    const int numVertices = static_cast<int>(DIM) + 1;

    // Evaluate the objective function at each vertex of the simplex.
    std::vector<scalar_t> f(numVertices);
    std::vector<int> idx(numVertices);
    for (int i = 0; i < numVertices; ++i) {
      f[i] = function(simplex_.col(i));
      idx[i] = i;
    }

    // Sort indices so that f[idx[0]] is the best (lowest) function value.
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return f[a] < f[b]; });

    // Check for degeneracy: if the maximum distance between the best and other
    // vertices is too small, restart the simplex around the best vertex.
    scalar_t maxDist = 0;
    for (int i = 1; i < numVertices; ++i) {
      scalar_t dist = (simplex_.col(idx[i]) - simplex_.col(idx[0]))
                          .template lpNorm<Eigen::Infinity>();
      if (dist > maxDist) {
        maxDist = dist;
      }
    }
    if (maxDist < degenerate_tol_) {
      simplex_ = makeInitialSimplex(simplex_.col(idx[0]));
      // Recompute function values and re-sort indices.
      for (int i = 0; i < numVertices; ++i) {
        f[i] = function(simplex_.col(i));
        idx[i] = i;
      }
      std::sort(idx.begin(), idx.end(),
                [&](int a, int b) { return f[a] < f[b]; });
    }

    // Compute the centroid of the best DIM vertices (all except the worst).
    vector_t x_bar = vector_t::Zero(DIM);
    for (size_t i = 0; i < DIM; ++i) {
      x_bar += simplex_.col(idx[i]);
    }
    x_bar /= static_cast<scalar_t>(DIM);

    // Reflection: compute the reflected point.
    vector_t x_r = (1 + rho_) * x_bar - rho_ * simplex_.col(idx[DIM]);
    scalar_t f_r = function(x_r);

    if (f_r < f[idx[0]]) {
      // Expansion: if the reflection is the best so far, try to expand further.
      vector_t x_e =
          (1 + rho_ * xi_) * x_bar - rho_ * xi_ * simplex_.col(idx[DIM]);
      scalar_t f_e = function(x_e);
      if (f_e < f_r) {
        simplex_.col(idx[DIM]) = x_e;
      } else {
        simplex_.col(idx[DIM]) = x_r;
      }
    } else if (f_r < f[idx[DIM - 1]]) {
      // Accept the reflected point.
      simplex_.col(idx[DIM]) = x_r;
    } else {
      // Contraction:
      if (f_r < f[idx[DIM]]) {
        // Outside contraction.
        vector_t x_c = (1 + rho_ * gamma_) * x_bar -
                       rho_ * gamma_ * simplex_.col(idx[DIM]);
        scalar_t f_c = function(x_c);
        if (f_c <= f_r) {
          simplex_.col(idx[DIM]) = x_c;
        } else {
          shrink(simplex_, idx, f, function);
        }
      } else {
        // Inside contraction.
        vector_t x_c = (1 - gamma_) * x_bar + gamma_ * simplex_.col(idx[DIM]);
        scalar_t f_c = function(x_c);
        if (f_c < f[idx[DIM]]) {
          simplex_.col(idx[DIM]) = x_c;
        } else {
          shrink(simplex_, idx, f, function);
        }
      }
    }

    // Return the current best vertex.
    return state_t(function, simplex_.col(idx[0]));
  }

 private:
  matrix_t simplex_;

  // Create an initial simplex given the starting point x using adaptive
  // perturbations.
  matrix_t makeInitialSimplex(const vector_t &x) {
    const size_t DIM = x.rows();
    matrix_t s = matrix_t::Zero(DIM, DIM + 1);
    for (size_t c = 0; c < DIM + 1; ++c) {
      for (size_t r = 0; r < DIM; ++r) {
        s(r, c) = x[r];
        if (r == c - 1) {
          // Use an adaptive perturbation for each coordinate.
          const scalar_t delta =
              (std::abs(x[r]) > 1e-6) ? 0.05 * std::abs(x[r]) : 0.001;
          s(r, c) += delta;
        }
      }
    }
    return s;
  }

  // Shrink the simplex toward the best vertex.
  void shrink(matrix_t &s, std::vector<int> &idx, std::vector<scalar_t> &f,
              const function_t &function) {
    const size_t DIM = s.rows();
    // Re-evaluate the best vertex.
    f[idx[0]] = function(s.col(idx[0]));
    for (size_t i = 1; i < DIM + 1; ++i) {
      s.col(idx[i]) = sigma_ * s.col(idx[i]) + (1 - sigma_) * s.col(idx[0]);
      f[idx[i]] = function(s.col(idx[i]));
    }
  }
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_NELDER_MEAD_H_
