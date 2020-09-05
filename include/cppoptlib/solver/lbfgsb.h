// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_
#define INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "../linesearch/more_thuente.h"
#include "Eigen/Core"
#include "solver.h"

namespace cppoptlib {
namespace solver {

namespace internal {};  // namespace internal

template <typename function_t, int m = 10>
class Lbfgsb : public Solver<function_t, 1> {
 public:
  using Superclass = Solver<function_t, 1>;
  using typename Superclass::state_t;
  using typename Superclass::scalar_t;
  using typename Superclass::hessian_t;
  using typename Superclass::matrix_t;
  using typename Superclass::vector_t;
  using typename Superclass::function_state_t;

  using dyn_vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;

  void InitializeSolver(const function_state_t &initial_state) override {
    theta_ = 1.0;

    W_ = matrix_t::Zero(function_t::Dim, 0);
    M_ = matrix_t::Zero(0, 0);

    yHistory_ = matrix_t::Zero(function_t::Dim, 0);
    sHistory_ = matrix_t::Zero(function_t::Dim, 0);
  }

  function_state_t OptimizationStep(const function_t &function,
                                    const function_state_t &current,
                                    const state_t &state) override {
    const vector_t upper_bound =
        current.x.array() * 0 + std::numeric_limits<scalar_t>::max();
    const vector_t lower_bound =
        current.x.array() * 0 + std::numeric_limits<scalar_t>::lowest();
    const vector_t &x = current.x;
    const vector_t &g = current.gradient;
    const scalar_t &f = current.value;

    // STEP 2: compute the cauchy point
    vector_t CauchyPoint = vector_t::Zero(function_t::Dim);
    dyn_vector_t c = dyn_vector_t::Zero(W_.cols());
    GetGeneralizedCauchyPoint(upper_bound, lower_bound, x, g, &CauchyPoint, &c);

    // STEP 3: compute a search direction d_k by the primal method for the
    // sub-problem
    const vector_t SubspaceMin =
        SubspaceMinimization(upper_bound, lower_bound, CauchyPoint, x, c, g);

    // STEP 4: perform linesearch and STEP 5: compute gradient
    scalar_t alpha_init = 1.0;
    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
        x, SubspaceMin - x, function, alpha_init);

    // update current guess and function information
    vector_t x_next = x - rate * (x - SubspaceMin);
    // if current solution is out of bound, we clip it
    ClampToBound(upper_bound, lower_bound, &x_next);

    function_state_t next = function.Eval(x_next);

    // prepare for next iteration
    const vector_t newY = next.gradient - current.gradient;
    const vector_t newS = next.x - current.x;

    // STEP 6:
    const scalar_t test = fabs(newS.dot(newY));
    if (test > 1e-7 * newY.squaredNorm()) {
      if (yHistory_.cols() < m_historySize) {
        yHistory_.conservativeResize(function_t::Dim, yHistory_.cols() + 1);
        sHistory_.conservativeResize(function_t::Dim, sHistory_.cols() + 1);
      } else {
        yHistory_.leftCols(m_historySize - 1) =
            yHistory_.rightCols(m_historySize - 1).eval();
        sHistory_.leftCols(m_historySize - 1) =
            sHistory_.rightCols(m_historySize - 1).eval();
      }
      yHistory_.rightCols(1) = newY;
      sHistory_.rightCols(1) = newS;
      // STEP 7:
      theta_ = (scalar_t)(newY.transpose() * newY) / (newY.transpose() * newS);
      W_ =
          matrix_t::Zero(yHistory_.rows(), yHistory_.cols() + sHistory_.cols());
      W_ << yHistory_, (theta_ * sHistory_);
      matrix_t A = sHistory_.transpose() * yHistory_;
      matrix_t L = A.template triangularView<Eigen::StrictlyLower>();
      matrix_t MM(A.rows() + L.rows(), A.rows() + L.cols());
      matrix_t D = -1 * A.diagonal().asDiagonal();
      MM << D, L.transpose(), L, ((sHistory_.transpose() * sHistory_) * theta_);
      M_ = MM.inverse();
    }

    return next;
  }

 private:
  /**
   * @brief sort pairs (k,v) according v ascending
   */
  std::vector<int> sort_indexes(
      const std::vector<std::pair<int, scalar_t> > &v) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = v[i].first;
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) { return v[i1].second < v[i2].second; });
    return idx;
  }

  void ClampToBound(const vector_t &upper_bound, const vector_t &lower_bound,
                    vector_t *x) {
    *x = x->cwiseMin(upper_bound).cwiseMax(lower_bound);

    // for (int r = 0; r < x->rows(); ++r) {
    //   if ((*x)(r) < lower_bound(r))
    //     (*x)(r) = lower_bound(r);
    //   else if ((*x)(r) > upper_bound(r))
    //     (*x)(r) = upper_bound(r);
    // }
  }

  void GetGeneralizedCauchyPoint(const vector_t &upper_bound,
                                 const vector_t &lower_bound, const vector_t &x,
                                 const vector_t &g, vector_t *x_cauchy,
                                 dyn_vector_t *c) {
    constexpr scalar_t max_value = std::numeric_limits<scalar_t>::max();
    constexpr scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon();

    const int DIM = x.rows();
    // Given x,l,u,g, and B = \theta_ I-WMW
    // {all t_i} = { (idx,value), ... }
    // TODO(patwie): use "std::set" ?
    std::vector<std::pair<int, scalar_t> > set_of_t;
    // the feasible set is implicitly given by "set_of_t - {t_i==0}"
    vector_t d = -g;
    // n operations
    for (int j = 0; j < DIM; j++) {
      if (g(j) == 0) {
        set_of_t.push_back(std::make_pair(j, max_value));
      } else {
        scalar_t tmp = 0;
        if (g(j) < 0) {
          tmp = (x(j) - upper_bound(j)) / g(j);
        } else {
          tmp = (x(j) - lower_bound(j)) / g(j);
        }
        set_of_t.push_back(std::make_pair(j, tmp));
        if (tmp == 0) d(j) = 0;
      }
    }
    // sortedindices [1,0,2] means the minimal element is on the 1-st entry
    std::vector<int> sorted_indices = sort_indexes(set_of_t);
    *x_cauchy = x;
    // Initialize
    // p :=     W^scalar_t*p
    dyn_vector_t p = (W_.transpose() * d);  // (2mn operations)
    // c :=     0
    *c = dyn_vector_t::Zero(W_.cols());
    // f' :=    g^scalar_t*d = -d^Td
    scalar_t f_prime = -d.dot(d);  // (n operations)
    // f'' :=   \theta_*d^scalar_t*d-d^scalar_t*W*M*W^scalar_t*d = -\theta_*f'
    // -
    // p^scalar_t*M*p
    scalar_t f_doubleprime = (scalar_t)(-1.0 * theta_) * f_prime -
                             p.dot(M_ * p);  // (O(m^2) operations)
    f_doubleprime = std::max<scalar_t>(epsilon, f_doubleprime);
    scalar_t f_dp_orig = f_doubleprime;
    // \delta t_min :=  -f'/f''
    scalar_t dt_min = -f_prime / f_doubleprime;
    // t_old :=     0
    scalar_t t_old = 0;
    // b :=     argmin {t_i , t_i >0}
    int i = 0;
    for (int j = 0; j < DIM; j++) {
      i = j;
      if (set_of_t[sorted_indices[j]].second > 0) break;
    }
    int b = sorted_indices[i];
    // see below
    // t                    :=  min{t_i : i in F}
    scalar_t t = set_of_t[b].second;
    // \delta scalar_t             :=  t - 0
    scalar_t dt = t;
    // examination of subsequent segments
    while ((dt_min >= dt) && (i < DIM)) {
      if (d(b) > 0)
        (*x_cauchy)(b) = upper_bound(b);
      else if (d(b) < 0)
        (*x_cauchy)(b) = lower_bound(b);
      // z_b = x_p^{cp} - x_b
      scalar_t zb = (*x_cauchy)(b)-x(b);
      // c   :=  c +\delta t*p
      *c += dt * p;
      // cache
      dyn_vector_t wbt = W_.row(b);
      f_prime += dt * f_doubleprime + g(b) * g(b) + theta_ * g(b) * zb -
                 g(b) * wbt.transpose() * (M_ * *c);
      f_doubleprime += (scalar_t)-1.0 * theta_ * g(b) * g(b) -
                       (scalar_t)2.0 * (g(b) * (wbt.dot(M_ * p))) -
                       (scalar_t)g(b) * g(b) * wbt.transpose() * (M_ * wbt);
      f_doubleprime = std::max<scalar_t>(
          std::numeric_limits<scalar_t>::epsilon() * f_dp_orig, f_doubleprime);
      p += g(b) * wbt.transpose();
      d(b) = 0;
      dt_min = -f_prime / f_doubleprime;
      t_old = t;
      ++i;
      if (i < DIM) {
        b = sorted_indices[i];
        t = set_of_t[b].second;
        dt = t - t_old;
      }
    }
    dt_min = std::max<scalar_t>(dt_min, (scalar_t)0.0);
    t_old += dt_min;
    // #pragma omp parallel for
    for (int ii = i; ii < x_cauchy->rows(); ii++) {
      (*x_cauchy)(sorted_indices[ii]) =
          x(sorted_indices[ii]) + t_old * d(sorted_indices[ii]);
    }
    *c += dt_min * p;
  }

  /**
   * @brief find alpha* = max {a : a <= 1 and  l_i-xc_i <= a*d_i <= u_i-xc_i}
   */
  scalar_t find_alpha(const vector_t &upper_bound, const vector_t &lower_bound,
                      const vector_t &x_cp, const dyn_vector_t &du,
                      std::vector<int> *FreeVariables) {
    scalar_t alphastar = 1;
    const unsigned int n = FreeVariables->size();
    assert(du.rows() == n);
    for (unsigned int i = 0; i < n; i++) {
      if (du(i) > 0) {
        alphastar = std::min<scalar_t>(
            alphastar,
            (upper_bound(FreeVariables->at(i)) - x_cp(FreeVariables->at(i))) /
                du(i));
      } else {
        alphastar = std::min<scalar_t>(
            alphastar,
            (lower_bound(FreeVariables->at(i)) - x_cp(FreeVariables->at(i))) /
                du(i));
      }
    }
    return alphastar;
  }

  vector_t SubspaceMinimization(const vector_t &upper_bound,
                                const vector_t &lower_bound,
                                const vector_t &x_cauchy, const vector_t &x,
                                const dyn_vector_t &c, const vector_t &g) {
    const scalar_t theta_inverse = 1 / theta_;
    std::vector<int> FreeVariablesIndex;
    for (int i = 0; i < x_cauchy.rows(); i++) {
      if ((x_cauchy(i) != upper_bound(i)) && (x_cauchy(i) != lower_bound(i))) {
        FreeVariablesIndex.push_back(i);
      }
    }
    const int FreeVarCount = FreeVariablesIndex.size();
    matrix_t WZ = matrix_t::Zero(W_.cols(), FreeVarCount);
    for (int i = 0; i < FreeVarCount; i++)
      WZ.col(i) = W_.row(FreeVariablesIndex[i]);
    vector_t rr = (g + theta_ * (x_cauchy - x) - W_ * (M_ * c));
    // r=r(FreeVariables);
    matrix_t r = matrix_t::Zero(FreeVarCount, 1);
    for (int i = 0; i < FreeVarCount; i++)
      r.row(i) = rr.row(FreeVariablesIndex[i]);
    // STEP 2: "v = w^T*Z*r" and STEP 3: "v = M*v"
    dyn_vector_t v = M_ * (WZ * r);
    // STEP 4: N = 1/theta*W^T*Z*(W^T*Z)^T
    matrix_t N = theta_inverse * WZ * WZ.transpose();
    // N = I - MN
    N = matrix_t::Identity(N.rows(), N.rows()) - M_ * N;
    // STEP: 5
    // v = N^{-1}*v
    if (v.size() > 0) v = N.lu().solve(v);
    // STEP: 6
    // HERE IS A MISTAKE IN THE ORIGINAL PAPER!
    dyn_vector_t du =
        -theta_inverse * r - theta_inverse * theta_inverse * WZ.transpose() * v;
    // STEP: 7
    scalar_t alpha_star =
        find_alpha(upper_bound, lower_bound, x_cauchy, du, &FreeVariablesIndex);
    // STEP: 8
    dyn_vector_t dStar = alpha_star * du;
    vector_t SubspaceMin = x_cauchy.eval();
    for (int i = 0; i < FreeVarCount; i++) {
      SubspaceMin(FreeVariablesIndex[i]) =
          SubspaceMin(FreeVariablesIndex[i]) + dStar(i);
    }
  }

 private:
  matrix_t M_;
  matrix_t W_;
  scalar_t theta_;

  matrix_t yHistory_;
  matrix_t sHistory_;

  static constexpr int m_historySize = 5;
};

}  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_