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
#ifndef INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_
#define INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "Eigen/LU"
#include "solver.h"  // NOLINT

namespace cppoptlib::solver {

template <typename FunctionType, int m = 5>
class Lbfgsb
    : public Solver<FunctionType, typename cppoptlib::function::FunctionState<
                                      typename FunctionType::ScalarType,
                                      FunctionType::Dimension>> {
  static_assert(
      FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::First ||
          FunctionType::Differentiability ==
              cppoptlib::function::DifferentiabilityMode::Second,
      "L-BFGS-B only supports first- or second-order differentiable functions");

 public:
  using StateType = typename cppoptlib::function::FunctionState<
      typename FunctionType::ScalarType, FunctionType::Dimension>;
  using Superclass = Solver<FunctionType, StateType>;
  using ProgressType = typename Superclass::ProgressType;

  using ScalarType = typename FunctionType::ScalarType;
  using MatrixType = typename FunctionType::MatrixType;
  using VectorType = typename FunctionType::VectorType;

  using dyn_MatrixType =
      Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
  using dyn_VectorType = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Superclass::Superclass;

  void SetBounds(const VectorType& lower_bound, const VectorType& upper_bound) {
    lower_bound_ = lower_bound;
    upper_bound_ = upper_bound;
    bounds_initialized_ = true;
  }

  void InitializeSolver(const FunctionType& /*function*/,
                        const StateType& initial_state) override {
    dim_ = initial_state.x.rows();

    if (!bounds_initialized_) {
      lower_bound_ =
          VectorType::Constant(dim_, std::numeric_limits<ScalarType>::lowest());
      upper_bound_ =
          VectorType::Constant(dim_, std::numeric_limits<ScalarType>::max());
      bounds_initialized_ = true;
    }

    theta_ = 1.0;

    W_ = dyn_MatrixType::Zero(dim_, 0);
    // MM_lu_ will be initialized when first L-BFGS update occurs

    y_history_ = dyn_MatrixType::Zero(dim_, 0);
    s_history_ = dyn_MatrixType::Zero(dim_, 0);
  }

  StateType OptimizationStep(const FunctionType& function,
                             const StateType& current,
                             const ProgressType& /*progress*/) override {
    // Project current point to bounds (handles infeasible initial points)
    const VectorType x =
        current.x.cwiseMin(upper_bound_).cwiseMax(lower_bound_);

    // STEP 2: compute the cauchy point
    VectorType cauchy_point = VectorType::Zero(x.rows());

    VectorType current_gradient;
    function(x, &current_gradient);

    dyn_VectorType c = dyn_VectorType::Zero(W_.cols());
    GetGeneralizedCauchyPoint(x, current_gradient, &cauchy_point, &c);

    // STEP 3: compute a search direction d_k by the primal method for the
    // sub-problem
    const auto [subspace_min, do_line_search] =
        SubspaceMinimization(x, current_gradient, cauchy_point, c);

    // STEP 4: perform linesearch and STEP 5: compute gradient
    ScalarType rate = 1.0;
    if (do_line_search) {
      ScalarType alpha_init = 1.0;
      rate = linesearch::MoreThuente<FunctionType, 1>::Search(
          x, subspace_min - x, function, alpha_init);
    }

    // update current guess and function information
    const VectorType x_next = x - rate * (x - subspace_min);
    // if current solution is out of bound, we clip it
    const VectorType clipped_x_next =
        x_next.cwiseMin(upper_bound_).cwiseMax(lower_bound_);

    const StateType next = StateType(clipped_x_next);
    VectorType next_gradient;
    function(next.x, &next_gradient);

    // prepare for next iteration
    const VectorType new_y = next_gradient - current_gradient;
    const VectorType new_s = next.x - x;

    // STEP 6: Only update if positive curvature (s'*y > 0)
    const ScalarType sTy = new_s.dot(new_y);
    if (sTy > 1e-7 * new_y.squaredNorm()) {
      if (y_history_.cols() < m) {
        y_history_.conservativeResize(dim_, y_history_.cols() + 1);
        s_history_.conservativeResize(dim_, s_history_.cols() + 1);
      } else {
        y_history_.leftCols(m - 1) = y_history_.rightCols(m - 1).eval();
        s_history_.leftCols(m - 1) = s_history_.rightCols(m - 1).eval();
      }
      y_history_.rightCols(1) = new_y;
      s_history_.rightCols(1) = new_s;
      // STEP 7:
      theta_ =
          (ScalarType)(new_y.transpose() * new_y) / (new_y.transpose() * new_s);
      W_ = dyn_MatrixType::Zero(y_history_.rows(),
                                y_history_.cols() + s_history_.cols());
      W_ << y_history_, (theta_ * s_history_);
      dyn_MatrixType A = s_history_.transpose() * y_history_;
      dyn_MatrixType L = A.template triangularView<Eigen::StrictlyLower>();
      dyn_MatrixType MM(A.rows() + L.rows(), A.rows() + L.cols());
      dyn_MatrixType D = -1 * A.diagonal().asDiagonal();
      MM << D, L.transpose(), L,
          ((s_history_.transpose() * s_history_) * theta_);
      // Store LU factorization for efficient triangular solves
      MM_lu_ = MM.lu();
    }

    return next;
  }

 private:
  /**
   * @brief sort pairs (k,v) according v ascending
   */
  static std::vector<int> SortIndexes(
      const std::vector<std::pair<int, ScalarType>>& v) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = v[i].first;
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) { return v[i1].second < v[i2].second; });
    return idx;
  }

  /**
   * @brief Solve MM * x = b using stored LU factorization (triangular solves)
   * More efficient than computing M_ * b when M_ = MM^{-1}
   */
  dyn_VectorType SolveM(const dyn_VectorType& b) const {
    if (b.size() == 0 || MM_lu_.matrixLU().size() == 0) {
      return b;
    }
    return MM_lu_.solve(b);
  }

  void GetGeneralizedCauchyPoint(const VectorType& x,
                                 const VectorType& gradient,
                                 VectorType* x_cauchy,
                                 dyn_VectorType* c) const {
    constexpr ScalarType max_value = std::numeric_limits<ScalarType>::max();
    // Use larger epsilon for numerical stability
    constexpr ScalarType epsilon = 1e-12;

    // Given x,l,u,g, and B = \theta_ I-WMW
    // {all t_i} = { (idx,value), ... }
    // TODO(patwie): use "std::set" ?
    std::vector<std::pair<int, ScalarType>> set_of_t;
    set_of_t.reserve(dim_);
    // The feasible set is implicitly given by "set_of_t - {t_i==0}".
    VectorType d = -gradient;
    // n operations
    for (int j = 0; j < dim_; j++) {
      if (gradient(j) == 0) {
        set_of_t.emplace_back(j, max_value);
      } else {
        ScalarType tmp = 0;
        if (gradient(j) < 0) {
          tmp = (x(j) - upper_bound_(j)) / gradient(j);
        } else {
          tmp = (x(j) - lower_bound_(j)) / gradient(j);
        }
        set_of_t.emplace_back(j, tmp);
        if (tmp == 0) d(j) = 0;
      }
    }
    // sortedindices [1,0,2] means the minimal element is on the 1-st entry
    std::vector<int> sorted_indices = SortIndexes(set_of_t);
    *x_cauchy = x;
    // Initialize
    // p :=     W^ScalarType*p
    dyn_VectorType p = (W_.transpose() * d);  // (2mn operations)
    // c :=     0
    *c = dyn_VectorType::Zero(W_.cols());
    // f' :=    g^ScalarType*d = -d^Td
    ScalarType f_prime = -d.dot(d);  // (n operations)
    // f'' :=   \theta_*d^ScalarType*d-d^ScalarType*W*M*W^ScalarType*d =
    // -\theta_*f' - p^ScalarType*M*p
    // Use triangular solve: M*p means solve(MM, p)
    ScalarType f_doubleprime = (ScalarType)(-1.0 * theta_) * f_prime -
                               p.dot(SolveM(p));  // (O(m^2) operations)
    f_doubleprime = std::max<ScalarType>(epsilon, f_doubleprime);
    ScalarType f_dp_orig = f_doubleprime;
    // \delta t_min :=  -f'/f''
    ScalarType dt_min = -f_prime / f_doubleprime;
    // t_old :=     0
    ScalarType t_old = 0;
    // b :=     argmin {t_i , t_i >0}
    int i = 0;
    for (int j = 0; j < dim_; j++) {
      i = j;
      if (set_of_t[sorted_indices[j]].second > 0) break;
    }
    int b = sorted_indices[i];
    // see below
    // t                    :=  min{t_i : i in F}
    ScalarType t = set_of_t[b].second;
    // \delta ScalarType             :=  t - 0
    ScalarType dt = t;
    // examination of subsequent segments
    while ((dt_min >= dt) && (i < dim_)) {
      if (d(b) > 0)
        (*x_cauchy)(b) = upper_bound_(b);
      else if (d(b) < 0)
        (*x_cauchy)(b) = lower_bound_(b);
      // z_b = x_p^{cp} - x_b
      const ScalarType zb = (*x_cauchy)(b)-x(b);
      // c   :=  c +\delta t*p
      *c += dt * p;
      // cache
      dyn_VectorType wbt = W_.row(b);
      dyn_VectorType Mc = SolveM(*c);
      dyn_VectorType Mp = SolveM(p);
      dyn_VectorType Mwbt = SolveM(wbt);
      f_prime += dt * f_doubleprime + gradient(b) * gradient(b) +
                 theta_ * gradient(b) * zb - gradient(b) * wbt.transpose() * Mc;
      f_doubleprime += ScalarType(-1.0) * theta_ * gradient(b) * gradient(b) -
                       ScalarType(2.0) * (gradient(b) * (wbt.dot(Mp))) -
                       gradient(b) * gradient(b) * wbt.transpose() * Mwbt;
      f_doubleprime = std::max<ScalarType>(epsilon * f_dp_orig, f_doubleprime);
      p += gradient(b) * wbt.transpose();
      d(b) = 0;
      dt_min = -f_prime / f_doubleprime;
      t_old = t;
      ++i;
      if (i < dim_) {
        b = sorted_indices[i];
        t = set_of_t[b].second;
        dt = t - t_old;
      }
    }
    dt_min = std::max<ScalarType>(dt_min, ScalarType{0});
    t_old += dt_min;

    (*x_cauchy)(sorted_indices) = x(sorted_indices) + t_old * d(sorted_indices);

    *c += dt_min * p;
  }

  /**
   * @brief find alpha* = max {a : a <= 1 and  l_i-xc_i <= a*d_i <= u_i-xc_i}
   */
  ScalarType FindAlpha(const VectorType& x_cp, const dyn_VectorType& du,
                       const std::vector<int>& free_variables) const {
    ScalarType alphastar = 1;
    const unsigned int n = free_variables.size();
    assert(du.rows() == n);

    for (unsigned int i = 0; i < n; i++) {
      if (std::abs(du(i)) < 1e-7) {
        continue;
      } else if (du(i) > 0) {
        alphastar = std::min<ScalarType>(
            alphastar,
            (upper_bound_(free_variables.at(i)) - x_cp(free_variables.at(i))) /
                du(i));
      } else {
        alphastar = std::min<ScalarType>(
            alphastar,
            (lower_bound_(free_variables.at(i)) - x_cp(free_variables.at(i))) /
                du(i));
      }
    }
    return alphastar;
  }

  std::pair<VectorType, bool> SubspaceMinimization(
      const VectorType& x, const VectorType& gradient,
      const VectorType& x_cauchy, const dyn_VectorType& c) const {
    std::vector<int> free_variables_index;
    for (int i = 0; i < x_cauchy.rows(); i++) {
      if ((x_cauchy(i) != upper_bound_(i)) &&
          (x_cauchy(i) != lower_bound_(i))) {
        free_variables_index.push_back(i);
      }
    }
    const int free_var_count = free_variables_index.size();

    // Early return if no free variables - Cauchy point is optimal
    if (free_var_count == 0) {
      return {x_cauchy, false};
    }

    const ScalarType theta_inverse = 1 / theta_;
    const dyn_MatrixType WZ =
        W_(free_variables_index, Eigen::indexing::all).transpose();

    const VectorType rr = (gradient + theta_ * (x_cauchy - x) - W_ * SolveM(c));
    // r=r(free_variables);
    const dyn_VectorType r = rr(free_variables_index);

    // STEP 2: "v = w^T*Z*r" and STEP 3: "v = M*v"
    dyn_VectorType v = SolveM(WZ * r);
    // STEP 4: N = 1/theta*W^T*Z*(W^T*Z)^T
    dyn_MatrixType N = theta_inverse * WZ * WZ.transpose();
    // N = I - MN (solve M*X = N for each column, then compute I - X)
    if (N.cols() > 0) {
      dyn_MatrixType MN(N.rows(), N.cols());
      for (int col = 0; col < N.cols(); ++col) {
        MN.col(col) = SolveM(N.col(col));
      }
      N = dyn_MatrixType::Identity(N.rows(), N.rows()) - MN;
    }
    // STEP: 5
    // v = N^{-1}*v
    if (v.size() > 0) {
      v = N.lu().solve(v);
    }
    // STEP: 6
    // HERE IS A MISTAKE IN THE ORIGINAL PAPER!
    const dyn_VectorType du =
        -theta_inverse * r - theta_inverse * theta_inverse * WZ.transpose() * v;
    // STEP: 7
    const ScalarType alpha_star = FindAlpha(x_cauchy, du, free_variables_index);
    // STEP: 8
    dyn_VectorType dStar = alpha_star * du;
    VectorType subspace_min = x_cauchy.eval();
    for (int i = 0; i < free_var_count; i++) {
      subspace_min(free_variables_index[i]) =
          subspace_min(free_variables_index[i]) + dStar(i);
    }
    return {subspace_min, true};
  }

 private:
  int dim_;
  VectorType lower_bound_;
  VectorType upper_bound_;
  bool bounds_initialized_ = false;

  Eigen::PartialPivLU<dyn_MatrixType> MM_lu_;
  dyn_MatrixType W_;
  ScalarType theta_;

  dyn_MatrixType y_history_;
  dyn_MatrixType s_history_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_
