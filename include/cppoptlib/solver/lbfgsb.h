// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
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

template <typename function_t, int m = 5>
class Lbfgsb : public Solver<function_t> {
 private:
  using Superclass = Solver<function_t>;
  using state_t = typename Superclass::state_t;

  using scalar_t = typename function_t::scalar_t;
  using hessian_t = typename function_t::hessian_t;
  using matrix_t = typename function_t::matrix_t;
  using vector_t = typename function_t::vector_t;
  using function_state_t = typename function_t::state_t;

  using dyn_vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit Lbfgsb(const State<scalar_t> &stopping_state =
                      DefaultStoppingSolverState<scalar_t>(),
                  typename Superclass::callback_t step_callback =
                      GetDefaultStepCallback<scalar_t, vector_t, hessian_t>())
      : Solver<function_t>{stopping_state, std::move(step_callback)} {}

  void InitializeSolver(const function_state_t &initial_state) override {
    dim_ = initial_state.x.rows();

    theta_ = 1.0;

    W_ = matrix_t::Zero(dim_, 0);
    M_ = matrix_t::Zero(0, 0);

    y_history_ = matrix_t::Zero(dim_, 0);
    s_history_ = matrix_t::Zero(dim_, 0);
  }

  function_state_t OptimizationStep(const function_t &function,
                                    const function_state_t &current,
                                    const state_t & /*state*/) override {
    const vector_t upper_bound =
        current.x.array() * 0 + std::numeric_limits<scalar_t>::max();
    const vector_t lower_bound =
        current.x.array() * 0 + std::numeric_limits<scalar_t>::lowest();

    // STEP 2: compute the cauchy point
    vector_t cauchy_point = vector_t::Zero(dim_);
    dyn_vector_t c = dyn_vector_t::Zero(W_.cols());
    GetGeneralizedCauchyPoint(upper_bound, lower_bound, current, &cauchy_point,
                              &c);

    // STEP 3: compute a search direction d_k by the primal method for the
    // sub-problem
    const vector_t subspace_min = SubspaceMinimization(
        upper_bound, lower_bound, current, cauchy_point, c);

    // STEP 4: perform linesearch and STEP 5: compute gradient
    scalar_t alpha_init = 1.0;
    const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
        current.x, subspace_min - current.x, function, alpha_init);

    // update current guess and function information
    const vector_t x_next = current.x - rate * (current.x - subspace_min);
    // if current solution is out of bound, we clip it
    const vector_t clipped_x_next =
        x_next.cwiseMin(upper_bound).cwiseMax(lower_bound);

    const function_state_t next = function.Eval(clipped_x_next, 1);

    // prepare for next iteration
    const vector_t new_y = next.gradient - current.gradient;
    const vector_t new_s = next.x - current.x;

    // STEP 6:
    const scalar_t test = fabs(new_s.dot(new_y));
    if (test > 1e-7 * new_y.squaredNorm()) {
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
          (scalar_t)(new_y.transpose() * new_y) / (new_y.transpose() * new_s);
      W_ = matrix_t::Zero(y_history_.rows(),
                          y_history_.cols() + s_history_.cols());
      W_ << y_history_, (theta_ * s_history_);
      matrix_t A = s_history_.transpose() * y_history_;
      matrix_t L = A.template triangularView<Eigen::StrictlyLower>();
      matrix_t MM(A.rows() + L.rows(), A.rows() + L.cols());
      matrix_t D = -1 * A.diagonal().asDiagonal();
      MM << D, L.transpose(), L,
          ((s_history_.transpose() * s_history_) * theta_);
      M_ = MM.inverse();
    }

    return next;
  }

 private:
  /**
   * @brief sort pairs (k,v) according v ascending
   */
  std::vector<int> SortIndexes(const std::vector<std::pair<int, scalar_t>> &v) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = v[i].first;
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) { return v[i1].second < v[i2].second; });
    return idx;
  }

  void GetGeneralizedCauchyPoint(const vector_t &upper_bound,
                                 const vector_t &lower_bound,
                                 const function_state_t &current,
                                 vector_t *x_cauchy, dyn_vector_t *c) {
    constexpr scalar_t max_value = std::numeric_limits<scalar_t>::max();
    constexpr scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon();

    // Given x,l,u,g, and B = \theta_ I-WMW
    // {all t_i} = { (idx,value), ... }
    // TODO(patwie): use "std::set" ?
    std::vector<std::pair<int, scalar_t>> set_of_t;
    // The feasible set is implicitly given by "set_of_t - {t_i==0}".
    vector_t d = -current.gradient;
    // n operations
    for (int j = 0; j < dim_; j++) {
      if (current.gradient(j) == 0) {
        set_of_t.push_back(std::make_pair(j, max_value));
      } else {
        scalar_t tmp = 0;
        if (current.gradient(j) < 0) {
          tmp = (current.x(j) - upper_bound(j)) / current.gradient(j);
        } else {
          tmp = (current.x(j) - lower_bound(j)) / current.gradient(j);
        }
        set_of_t.push_back(std::make_pair(j, tmp));
        if (tmp == 0) d(j) = 0;
      }
    }
    // sortedindices [1,0,2] means the minimal element is on the 1-st entry
    std::vector<int> sorted_indices = SortIndexes(set_of_t);
    *x_cauchy = current.x;
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
    for (int j = 0; j < dim_; j++) {
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
    while ((dt_min >= dt) && (i < dim_)) {
      if (d(b) > 0)
        (*x_cauchy)(b) = upper_bound(b);
      else if (d(b) < 0)
        (*x_cauchy)(b) = lower_bound(b);
      // z_b = x_p^{cp} - x_b
      const scalar_t zb = (*x_cauchy)(b)-current.x(b);
      // c   :=  c +\delta t*p
      *c += dt * p;
      // cache
      dyn_vector_t wbt = W_.row(b);
      f_prime += dt * f_doubleprime +
                 current.gradient(b) * current.gradient(b) +
                 theta_ * current.gradient(b) * zb -
                 current.gradient(b) * wbt.transpose() * (M_ * *c);
      f_doubleprime +=
          scalar_t(-1.0) * theta_ * current.gradient(b) * current.gradient(b) -
          scalar_t(2.0) * (current.gradient(b) * (wbt.dot(M_ * p))) -
          current.gradient(b) * current.gradient(b) * wbt.transpose() *
              (M_ * wbt);
      f_doubleprime = std::max<scalar_t>(epsilon * f_dp_orig, f_doubleprime);
      p += current.gradient(b) * wbt.transpose();
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
    dt_min = std::max<scalar_t>(dt_min, scalar_t{0});
    t_old += dt_min;

    (*x_cauchy)(sorted_indices) =
        current.x(sorted_indices) + t_old * d(sorted_indices);

    *c += dt_min * p;
  }

  /**
   * @brief find alpha* = max {a : a <= 1 and  l_i-xc_i <= a*d_i <= u_i-xc_i}
   */
  scalar_t FindAlpha(const vector_t &upper_bound, const vector_t &lower_bound,
                     const vector_t &x_cp, const dyn_vector_t &du,
                     const std::vector<int> &free_variables) {
    scalar_t alphastar = 1;
    const unsigned int n = free_variables.size();
    assert(du.rows() == n);

    for (unsigned int i = 0; i < n; i++) {
      if (du(i) > 0) {
        alphastar = std::min<scalar_t>(
            alphastar,
            (upper_bound(free_variables.at(i)) - x_cp(free_variables.at(i))) /
                du(i));
      } else {
        alphastar = std::min<scalar_t>(
            alphastar,
            (lower_bound(free_variables.at(i)) - x_cp(free_variables.at(i))) /
                du(i));
      }
    }
    return alphastar;
  }

  vector_t SubspaceMinimization(const vector_t &upper_bound,
                                const vector_t &lower_bound,
                                const function_state_t &current,
                                const vector_t &x_cauchy,
                                const dyn_vector_t &c) {
    const scalar_t theta_inverse = 1 / theta_;

    std::vector<int> free_variables_index;
    for (int i = 0; i < x_cauchy.rows(); i++) {
      if ((x_cauchy(i) != upper_bound(i)) && (x_cauchy(i) != lower_bound(i))) {
        free_variables_index.push_back(i);
      }
    }
    const int free_var_count = free_variables_index.size();
    const matrix_t WZ =
        W_(free_variables_index, Eigen::indexing::all).transpose();

    const vector_t rr =
        (current.gradient + theta_ * (x_cauchy - current.x) - W_ * (M_ * c));
    // r=r(free_variables);
    const matrix_t r = rr(free_variables_index);

    // STEP 2: "v = w^T*Z*r" and STEP 3: "v = M*v"
    dyn_vector_t v = M_ * (WZ * r);
    // STEP 4: N = 1/theta*W^T*Z*(W^T*Z)^T
    matrix_t N = theta_inverse * WZ * WZ.transpose();
    // N = I - MN
    N = matrix_t::Identity(N.rows(), N.rows()) - M_ * N;
    // STEP: 5
    // v = N^{-1}*v
    if (v.size() > 0) {
      v = N.lu().solve(v);
    }
    // STEP: 6
    // HERE IS A MISTAKE IN THE ORIGINAL PAPER!
    const dyn_vector_t du =
        -theta_inverse * r - theta_inverse * theta_inverse * WZ.transpose() * v;
    // STEP: 7
    const scalar_t alpha_star =
        FindAlpha(upper_bound, lower_bound, x_cauchy, du, free_variables_index);
    // STEP: 8
    dyn_vector_t dStar = alpha_star * du;
    vector_t subspace_min = x_cauchy.eval();
    for (int i = 0; i < free_var_count; i++) {
      subspace_min(free_variables_index[i]) =
          subspace_min(free_variables_index[i]) + dStar(i);
    }
    return subspace_min;
  }

 private:
  int dim_;

  matrix_t M_;
  matrix_t W_;
  scalar_t theta_;

  matrix_t y_history_;
  matrix_t s_history_;
};

}  // namespace cppoptlib::solver

#endif  // INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_
