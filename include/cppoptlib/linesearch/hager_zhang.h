// CPPNumericalSolvers - A lightweight C++ numerical optimization library
// Copyright (c) 2026    Patrick Wieschollek + Contributors
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
#ifndef INCLUDE_CPPOPTLIB_LINESEARCH_HAGER_ZHANG_H_
#define INCLUDE_CPPOPTLIB_LINESEARCH_HAGER_ZHANG_H_

// Hager-Zhang (2006) line search.
//
// Reference: W. W. Hager and H. Zhang, "Algorithm 851: CG_DESCENT, a conjugate
// gradient method with guaranteed descent", ACM TOMS 32(1), 2006, pp.113-137.
//
// This is a faithful port of `LineSearches.jl`'s `hagerzhang.jl` (v7.6.1).
// Comment tags such as "HZ, stage Bx" / "HZ, stage Ux" / "HZ, stage Sx" /
// "HZ, eq. N" refer verbatim to the paper sections and equations, preserved
// from the Julia source so a reviewer can cross-check step-by-step.
//
// The public surface mirrors `more_thuente.h` exactly (three `Search`
// overloads) so this class is a drop-in `LineSearch` template parameter for
// solvers such as `Lbfgsb`.

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "../function_base.h"

namespace cppoptlib::solver::linesearch {

template <typename FunctionType, int Ord>
class HagerZhang {
 public:
  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;

  /**
   * @brief Run the Hager-Zhang search and return only the step width.
   */
  static ScalarType Search(const VectorType& x,
                           const VectorType& search_direction,
                           const FunctionType& function,
                           const ScalarType alpha_init = 1.0) {
    VectorType g;
    ScalarType f = function(x, &g);
    VectorType s = search_direction.eval();
    ScalarType alpha = alpha_init;
    VectorType xx = x;
    hzls(function, &xx, &f, &g, &alpha, s);
    return alpha;
  }

  /**
   * @brief Run the Hager-Zhang search given a cached `(f0, g0)` at `x`.
   */
  static ScalarType Search(const VectorType& x, ScalarType f0,
                           const VectorType& g0,
                           const VectorType& search_direction,
                           const FunctionType& function, ScalarType alpha_init,
                           VectorType* x_out, ScalarType* f_out,
                           VectorType* g_out) {
    ScalarType alpha = alpha_init;
    ScalarType f = f0;
    VectorType g = g0;
    VectorType s = search_direction.eval();
    VectorType xx = x;
    hzls(function, &xx, &f, &g, &alpha, s);
    if (x_out) *x_out = std::move(xx);
    if (f_out) *f_out = f;
    if (g_out) *g_out = std::move(g);
    return alpha;
  }

  /**
   * @brief Run the Hager-Zhang search from a fully-evaluated starting state
   * and return a fully-evaluated accepted state.
   */
  template <class State>
  static State Search(const State& start, const VectorType& search_direction,
                      const FunctionType& function,
                      const ScalarType alpha_init = ScalarType(1),
                      ScalarType* alpha_out = nullptr) {
    ScalarType alpha = alpha_init;
    ScalarType f = start.value;
    VectorType g = start.gradient;
    VectorType s = search_direction.eval();
    VectorType xx = start.x;
    hzls(function, &xx, &f, &g, &alpha, s);
    if (alpha_out) *alpha_out = alpha;
    return State(std::move(xx), f, std::move(g));
  }

 private:
  // Helper result-slot for the scalar line-search history and the "best"
  // state to return.  We reuse the slot so we never re-allocate `VectorType`
  // inside the loop; the final accepted `(x, f, g)` is written through the
  // caller-provided pointers.
  struct EvalResult {
    ScalarType alpha;
    ScalarType phi;
    ScalarType dphi;
  };

  // HZ Wolfe and approximate-Wolfe acceptance test (paper equations T1/T2).
  // The approximate-Wolfe (T2) branch becomes active only once `phi_c`
  // drops within the epsilon envelope of `phi_0`, which is why we also
  // demand `phi_c <= phi_lim`.
  static bool SatisfiesWolfe(ScalarType c, ScalarType phi_c, ScalarType dphi_c,
                             ScalarType phi_0, ScalarType dphi_0,
                             ScalarType phi_lim, ScalarType delta,
                             ScalarType sigma) {
    const bool wolfe1 =
        (delta * dphi_0 >= (phi_c - phi_0) / c) && (dphi_c >= sigma * dphi_0);
    const bool wolfe2 = ((2 * delta - 1) * dphi_0 >= dphi_c) &&
                        (dphi_c >= sigma * dphi_0) && (phi_c <= phi_lim);
    return wolfe1 || wolfe2;
  }

  // HZ, stages S1-S4: scalar secant between two derivative samples.
  static ScalarType Secant(ScalarType a, ScalarType b, ScalarType dphi_a,
                           ScalarType dphi_b) {
    return (a * dphi_b - b * dphi_a) / (dphi_b - dphi_a);
  }

  // Evaluate phi(alpha) = f(x + alpha * s) and its directional derivative
  // dphi(alpha) = g(x+alpha*s)^T s, reusing workspace to avoid allocation.
  static void PhiDphi(const FunctionType& function, const VectorType& x,
                      const VectorType& s, ScalarType alpha, VectorType* xa,
                      VectorType* g, ScalarType* phi, ScalarType* dphi) {
    *xa = x + alpha * s;
    *phi = function(*xa, g);
    *dphi = g->dot(s);
  }

  // HZ, stages U0-U3.  Given a third point `c` with value/slope already in
  // `history[ic]`, shrink the bracket `[ia, ib]` to two indices of a
  // bracket around the minimum (HZ eq. 29).  Returns `(new_ia, new_ib,
  // wolfe_hit)`.
  static std::tuple<int, int, bool> Update(
      const FunctionType& function, const VectorType& x, const VectorType& s,
      std::vector<EvalResult>* history, VectorType* xa, VectorType* g, int ia,
      int ib, int ic, ScalarType phi_lim, ScalarType phi_0, ScalarType dphi_0,
      ScalarType delta, ScalarType sigma) {
    const ScalarType a = (*history)[ia].alpha;
    const ScalarType b = (*history)[ib].alpha;
    const ScalarType c = (*history)[ic].alpha;
    const ScalarType phi_c = (*history)[ic].phi;
    const ScalarType dphi_c = (*history)[ic].dphi;
    // HZ, U0: out-of-bracket points are rejected.
    if (c < a || c > b) return {ia, ib, false};
    // HZ, U1: upward slope means c is a new upper bound.
    if (dphi_c >= ScalarType(0)) return {ia, ic, false};
    // HZ, U2: downward slope with low value means c is a better lower bound.
    if (phi_c <= phi_lim) return {ic, ib, false};
    // HZ, U3: downward slope but value above phi_lim => minimum lies in
    // [a, c]; find it via bisection.
    return Bisect(function, x, s, history, xa, g, ia, ic, phi_lim, phi_0,
                  dphi_0, delta, sigma);
  }

  // HZ, stage U3 with theta = 0.5: bisect until slope flips or interval
  // collapses, keeping `a` on the descent side.
  static std::tuple<int, int, bool> Bisect(
      const FunctionType& function, const VectorType& x, const VectorType& s,
      std::vector<EvalResult>* history, VectorType* xa, VectorType* g, int ia,
      int ib, ScalarType phi_lim, ScalarType phi_0, ScalarType dphi_0,
      ScalarType delta, ScalarType sigma) {
    ScalarType a = (*history)[ia].alpha;
    ScalarType b = (*history)[ib].alpha;
    while (b - a > std::numeric_limits<ScalarType>::epsilon() * b) {
      const ScalarType d = (a + b) / ScalarType(2);
      EvalResult r;
      r.alpha = d;
      PhiDphi(function, x, s, d, xa, g, &r.phi, &r.dphi);
      history->push_back(r);
      const int id = static_cast<int>(history->size()) - 1;
      if (SatisfiesWolfe(d, r.phi, r.dphi, phi_0, dphi_0, phi_lim, delta,
                         sigma)) {
        return {ia, id, true};
      }
      if (r.dphi >= ScalarType(0)) return {ia, id, false};
      if (r.phi <= phi_lim) {
        a = d;
        ia = id;
      } else {
        b = d;
        ib = id;
      }
    }
    return {ia, ib, false};
  }

  // HZ, stages S1-S4: take two secant-driven candidates, call Update up to
  // twice.  Returns `(wolfe_hit, new_ia, new_ib)`.
  static std::tuple<bool, int, int> Secant2(
      const FunctionType& function, const VectorType& x, const VectorType& s,
      std::vector<EvalResult>* history, VectorType* xa, VectorType* g, int ia,
      int ib, ScalarType phi_lim, ScalarType phi_0, ScalarType dphi_0,
      ScalarType delta, ScalarType sigma) {
    const ScalarType a = (*history)[ia].alpha;
    const ScalarType b = (*history)[ib].alpha;
    const ScalarType dphi_a = (*history)[ia].dphi;
    const ScalarType dphi_b = (*history)[ib].dphi;
    // HZ, S1: first secant between the two bracket endpoints.
    ScalarType c = Secant(a, b, dphi_a, dphi_b);
    if (!std::isfinite(c)) c = (a + b) / ScalarType(2);
    EvalResult r;
    r.alpha = c;
    PhiDphi(function, x, s, c, xa, g, &r.phi, &r.dphi);
    history->push_back(r);
    const int ic = static_cast<int>(history->size()) - 1;
    if (SatisfiesWolfe(c, r.phi, r.dphi, phi_0, dphi_0, phi_lim, delta,
                       sigma)) {
      return {true, ic, ic};
    }
    // HZ, S2: update bracket with the secant sample.
    auto [iA, iB, wolfe_hit] = Update(function, x, s, history, xa, g, ia, ib,
                                      ic, phi_lim, phi_0, dphi_0, delta, sigma);
    if (wolfe_hit) return {true, iB, iB};
    const ScalarType A = (*history)[iA].alpha;
    const ScalarType B = (*history)[iB].alpha;
    // HZ, S3: if exactly one endpoint moved, take a second secant between
    // the old endpoint on that side and the updated one.
    ScalarType c2 = c;
    const bool moved_b = (iB == ic);
    const bool moved_a = (iA == ic);
    if (moved_b) {
      c2 = Secant((*history)[ib].alpha, (*history)[iB].alpha,
                  (*history)[ib].dphi, (*history)[iB].dphi);
    } else if (moved_a) {
      c2 = Secant((*history)[ia].alpha, (*history)[iA].alpha,
                  (*history)[ia].dphi, (*history)[iA].dphi);
    }
    if ((moved_a || moved_b) && A <= c2 && c2 <= B) {
      EvalResult r2;
      r2.alpha = c2;
      PhiDphi(function, x, s, c2, xa, g, &r2.phi, &r2.dphi);
      history->push_back(r2);
      const int ic2 = static_cast<int>(history->size()) - 1;
      if (SatisfiesWolfe(c2, r2.phi, r2.dphi, phi_0, dphi_0, phi_lim, delta,
                         sigma)) {
        return {true, ic2, ic2};
      }
      // HZ, S4: second update.
      auto [iA2, iB2, wolfe_hit2] =
          Update(function, x, s, history, xa, g, iA, iB, ic2, phi_lim, phi_0,
                 dphi_0, delta, sigma);
      if (wolfe_hit2) return {true, iB2, iB2};
      return {false, iA2, iB2};
    }
    return {false, iA, iB};
  }

  // Main driver.  `*x` carries the initial iterate on entry and the
  // accepted iterate on exit; `*f`, `*g` are filled with the accepted
  // function value and gradient; `*stp` carries the initial trial step and
  // the accepted step width.  Returns 0 on success, negative on failure
  // (no descent direction, or max-iter exhausted with no bracket).
  static int hzls(const FunctionType& function, VectorType* x, ScalarType* f,
                  VectorType* g, ScalarType* stp, const VectorType& s) {
    // HZ paper defaults -- do NOT expose a config struct (match
    // `more_thuente.h` style).
    constexpr ScalarType delta = ScalarType(1) / ScalarType(10);  // c_1
    constexpr ScalarType sigma = ScalarType(9) / ScalarType(10);  // c_2
    constexpr ScalarType epsilon_k = ScalarType(1e-6);            // HZ eps
    constexpr ScalarType gamma = ScalarType(0.66);                // shrink
    constexpr ScalarType rho = ScalarType(5);                     // expand
    constexpr ScalarType psi3 = ScalarType(0.1);                  // HZ I0
    constexpr int maxlinesearch = 50;
    // Reserve enough history up front: bracket phase + main loop each push
    // at most `maxlinesearch` evaluations, plus one for the initial step
    // and the origin.  A slack factor of 4 covers bisection inside
    // Update/Bisect.
    constexpr int history_reserve = 4 * maxlinesearch + 2;

    const ScalarType phi_0 = *f;
    const ScalarType dphi_0 = g->dot(s);
    // No-descent guard matches MoreThuente's behaviour.
    if (dphi_0 >= ScalarType(0)) return -1;

    const ScalarType phi_lim = phi_0 + epsilon_k * std::abs(phi_0);

    std::vector<EvalResult> history;
    history.reserve(history_reserve);
    history.push_back({ScalarType(0), phi_0, dphi_0});

    // Workspaces reused by every PhiDphi evaluation; never reallocated
    // inside the loop.
    VectorType xa = *x;
    VectorType gx = *g;

    // Best-seen sample, kept updated so we can return a descending step
    // even if the bracket loop bails early.
    ScalarType best_alpha = ScalarType(0);
    ScalarType best_phi = phi_0;
    VectorType best_x = *x;
    VectorType best_g = *g;

    auto update_best = [&](ScalarType alpha, ScalarType phi_val) {
      if (alpha > ScalarType(0) && phi_val < best_phi) {
        best_alpha = alpha;
        best_phi = phi_val;
        best_x = xa;
        best_g = gx;
      }
    };

    // Initial trial step.  Guard against alpha_init <= 0 which would
    // break the division in SatisfiesWolfe.
    ScalarType c = *stp;
    if (!(c > ScalarType(0))) c = ScalarType(1);

    EvalResult ec;
    ec.alpha = c;
    PhiDphi(function, *x, s, c, &xa, &gx, &ec.phi, &ec.dphi);
    // HZ, I0: if the trial point is non-finite, shrink by psi3 a few times
    // before giving up.  This can happen when the solver takes a step that
    // overshoots into a region with overflow.
    int iterfinite = 0;
    constexpr int iterfinitemax = 60;  // ceil(-log2(eps<double>))
    while (!(std::isfinite(ec.phi) && std::isfinite(ec.dphi)) &&
           iterfinite < iterfinitemax) {
      c *= psi3;
      ec.alpha = c;
      PhiDphi(function, *x, s, c, &xa, &gx, &ec.phi, &ec.dphi);
      ++iterfinite;
    }
    if (!(std::isfinite(ec.phi) && std::isfinite(ec.dphi))) {
      // Give up: return alpha=0 with the starting state untouched.
      *stp = ScalarType(0);
      return -1;
    }
    history.push_back(ec);
    update_best(c, ec.phi);
    if (SatisfiesWolfe(c, ec.phi, ec.dphi, phi_0, dphi_0, phi_lim, delta,
                       sigma)) {
      *x = xa;
      *f = ec.phi;
      *g = gx;
      *stp = c;
      return 0;
    }

    // HZ, stages B0-B3: expand until we find a bracket `[a, b]` with
    // `dphi(a) < 0 <= dphi(b)` and `phi(a) <= phi_lim`.
    bool bracketed = false;
    int ia = 0;
    int ib = 1;
    int iter = 1;
    while (!bracketed && iter < maxlinesearch) {
      const EvalResult& last = history.back();
      if (last.dphi >= ScalarType(0)) {
        // HZ, B1: upward slope -> we have `b`; scan back for a feasible a.
        ib = static_cast<int>(history.size()) - 1;
        for (int i = ib - 1; i >= 0; --i) {
          if (history[i].phi <= phi_lim) {
            ia = i;
            break;
          }
        }
        bracketed = true;
      } else if (last.phi > phi_lim) {
        // HZ, B2: downward slope but climbed over a peak -> bisect.
        ib = static_cast<int>(history.size()) - 1;
        ia = 0;
        auto [new_ia, new_ib, wolfe_hit] =
            Bisect(function, *x, s, &history, &xa, &gx, ia, ib, phi_lim, phi_0,
                   dphi_0, delta, sigma);
        if (wolfe_hit) {
          const EvalResult& w = history[new_ib];
          *x = xa;
          *f = w.phi;
          *g = gx;
          *stp = w.alpha;
          return 0;
        }
        ia = new_ia;
        ib = new_ib;
        bracketed = true;
      } else {
        // HZ, B3: still descending, expand.  Remember the previous point
        // as the best-known fallback.
        c *= rho;
        ec.alpha = c;
        PhiDphi(function, *x, s, c, &xa, &gx, &ec.phi, &ec.dphi);
        iterfinite = 0;
        while (!(std::isfinite(ec.phi) && std::isfinite(ec.dphi)) &&
               iterfinite < iterfinitemax) {
          // Bisect back toward the previous finite point.
          c = (history.back().alpha + c) / ScalarType(2);
          ec.alpha = c;
          PhiDphi(function, *x, s, c, &xa, &gx, &ec.phi, &ec.dphi);
          ++iterfinite;
        }
        if (!(std::isfinite(ec.phi) && std::isfinite(ec.dphi))) {
          // Fall back to best known.
          if (best_alpha > ScalarType(0)) {
            *x = best_x;
            *f = best_phi;
            *g = best_g;
            *stp = best_alpha;
            return 0;
          }
          *stp = ScalarType(0);
          return -1;
        }
        history.push_back(ec);
        update_best(c, ec.phi);
        if (SatisfiesWolfe(c, ec.phi, ec.dphi, phi_0, dphi_0, phi_lim, delta,
                           sigma)) {
          *x = xa;
          *f = ec.phi;
          *g = gx;
          *stp = c;
          return 0;
        }
      }
      ++iter;
    }

    if (!bracketed) {
      // Bail with the best seen point (still a descent step).
      if (best_alpha > ScalarType(0)) {
        *x = best_x;
        *f = best_phi;
        *g = best_g;
        *stp = best_alpha;
        return 0;
      }
      *stp = ScalarType(0);
      return -1;
    }

    // Main shrinking loop.  Invariant: `history[ia]` and `history[ib]`
    // form a bracket around a minimum.
    while (iter < maxlinesearch) {
      const ScalarType a = history[ia].alpha;
      const ScalarType b = history[ib].alpha;
      if (b - a <= std::numeric_limits<ScalarType>::epsilon() * b) {
        // Interval collapsed; return the a-side (guaranteed feasible).
        // Recompute `xa, gx` at `a` so the output is self-consistent.
        if (a > ScalarType(0)) {
          PhiDphi(function, *x, s, a, &xa, &gx, &ec.phi, &ec.dphi);
          *x = xa;
          *f = ec.phi;
          *g = gx;
          *stp = a;
          return 0;
        }
        // a == 0 is degenerate; return best-seen.
        if (best_alpha > ScalarType(0)) {
          *x = best_x;
          *f = best_phi;
          *g = best_g;
          *stp = best_alpha;
          return 0;
        }
        *stp = ScalarType(0);
        return -1;
      }
      auto [wolfe_hit, iA, iB] =
          Secant2(function, *x, s, &history, &xa, &gx, ia, ib, phi_lim, phi_0,
                  dphi_0, delta, sigma);
      if (wolfe_hit) {
        const EvalResult& w = history[iA];
        // `xa`/`gx` hold the last evaluation, which for a Wolfe hit is at
        // `history[iA].alpha` (Secant2 writes `iA == iB` in that path).
        *x = xa;
        *f = w.phi;
        *g = gx;
        *stp = w.alpha;
        return 0;
      }
      const ScalarType A = history[iA].alpha;
      const ScalarType B = history[iB].alpha;
      if (B - A < gamma * (b - a)) {
        // Good shrink; take it.
        ia = iA;
        ib = iB;
      } else {
        // HZ, stage L2: secant too slow, fall back to bisection.
        const ScalarType cm = (A + B) / ScalarType(2);
        EvalResult rm;
        rm.alpha = cm;
        PhiDphi(function, *x, s, cm, &xa, &gx, &rm.phi, &rm.dphi);
        history.push_back(rm);
        const int ic = static_cast<int>(history.size()) - 1;
        update_best(cm, rm.phi);
        if (SatisfiesWolfe(cm, rm.phi, rm.dphi, phi_0, dphi_0, phi_lim, delta,
                           sigma)) {
          *x = xa;
          *f = rm.phi;
          *g = gx;
          *stp = cm;
          return 0;
        }
        auto [new_ia, new_ib, wolfe_hit2] =
            Update(function, *x, s, &history, &xa, &gx, iA, iB, ic, phi_lim,
                   phi_0, dphi_0, delta, sigma);
        if (wolfe_hit2) {
          const EvalResult& w = history[new_ib];
          *x = xa;
          *f = w.phi;
          *g = gx;
          *stp = w.alpha;
          return 0;
        }
        ia = new_ia;
        ib = new_ib;
      }
      ++iter;
    }

    // Max iterations hit without a Wolfe point: return best-seen.
    if (best_alpha > ScalarType(0)) {
      *x = best_x;
      *f = best_phi;
      *g = best_g;
      *stp = best_alpha;
      return 0;
    }
    *stp = ScalarType(0);
    return -1;
  }
};

}  // namespace cppoptlib::solver::linesearch

#endif  // INCLUDE_CPPOPTLIB_LINESEARCH_HAGER_ZHANG_H_
