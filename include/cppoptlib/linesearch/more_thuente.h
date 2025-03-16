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
#ifndef INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_
#define INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_

#include <algorithm>
#include <cmath>

namespace cppoptlib::solver::linesearch {

template <typename FunctionType, int Ord>
class MoreThuente {
 public:
  using ScalarType = typename FunctionType::ScalarType;
  using VectorType = typename FunctionType::VectorType;

  /**
   * @brief use MoreThuente Rule for (strong) Wolfe conditiions
   * @details [long description]
   *
   * @param search_direction search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */

  static ScalarType Search(const VectorType &x,
                           const VectorType &search_direction,
                           const FunctionType &function,
                           const ScalarType alpha_init = 1.0) {
    ScalarType alpha = alpha_init;
    VectorType g;
    const ScalarType v = function(x, &g);

    VectorType s = search_direction.eval();
    VectorType xx = x;

    cvsrch(function, &xx, v, &g, &alpha, s);

    return alpha;
  }

  static int cvsrch(const FunctionType &function, VectorType *x, ScalarType f,
                    VectorType *g, ScalarType *stp, const VectorType &s) {
    // we rewrite this from MIN-LAPACK and some MATLAB code
    int info = 0;
    int infoc = 1;
    constexpr ScalarType xtol = 1e-15;
    constexpr ScalarType ftol = 1e-4;
    constexpr ScalarType gtol = 1e-2;
    constexpr ScalarType stpmin = 1e-15;
    constexpr ScalarType stpmax = 1e15;
    constexpr ScalarType xtrapf = 4;
    constexpr int maxfev = 20;
    int nfev = 0;

    ScalarType dginit = g->dot(s);
    if (dginit >= 0.0) {
      // There is no descent direction.
      // TODO(patwie): Handle this case.
      return -1;
    }

    bool brackt = false;
    bool stage1 = true;

    ScalarType finit = f;
    ScalarType dgtest = ftol * dginit;
    ScalarType width = stpmax - stpmin;
    ScalarType width1 = 2 * width;
    VectorType wa = x->eval();

    ScalarType stx = 0.0;
    ScalarType fx = finit;
    ScalarType dgx = dginit;
    ScalarType sty = 0.0;
    ScalarType fy = finit;
    ScalarType dgy = dginit;

    ScalarType stmin;
    ScalarType stmax;

    while (true) {
      // Make sure we stay in the interval when setting min/max-step-width.
      if (brackt) {
        stmin = std::min<ScalarType>(stx, sty);
        stmax = std::max<ScalarType>(stx, sty);
      } else {
        stmin = stx;
        stmax = *stp + xtrapf * (*stp - stx);
      }

      // Force the step to be within the bounds stpmax and stpmin.
      *stp = std::clamp(*stp, stpmin, stpmax);

      // Oops, let us return the last reliable values.
      if ((brackt && ((*stp <= stmin) || (*stp >= stmax))) ||
          (nfev >= maxfev - 1) || (infoc == 0) ||
          (brackt && ((stmax - stmin) <= (xtol * stmax)))) {
        *stp = stx;
      }

      // Test new point.
      *x = wa + *stp * s;
      f = function(*x, g);
      nfev++;
      ScalarType dg = g->dot(s);
      ScalarType ftest1 = finit + *stp * dgtest;

      // All possible convergence tests.
      if ((brackt & ((*stp <= stmin) | (*stp >= stmax))) | (infoc == 0))
        info = 6;

      if ((*stp == stpmax) & (f <= ftest1) & (dg <= dgtest)) info = 5;

      if ((*stp == stpmin) & ((f > ftest1) | (dg >= dgtest))) info = 4;

      if (nfev >= maxfev) info = 3;

      if (brackt & (stmax - stmin <= xtol * stmax)) info = 2;

      if ((f <= ftest1) & (fabs(dg) <= gtol * (-dginit))) info = 1;

      // Terminate when convergence reached.
      if (info != 0) return -1;

      if (stage1 & (f <= ftest1) &
          (dg >= std::min<ScalarType>(ftol, gtol) * dginit))
        stage1 = false;

      if (stage1 & (f <= fx) & (f > ftest1)) {
        ScalarType fm = f - *stp * dgtest;
        ScalarType fxm = fx - stx * dgtest;
        ScalarType fym = fy - sty * dgtest;
        ScalarType dgm = dg - dgtest;
        ScalarType dgxm = dgx - dgtest;
        ScalarType dgym = dgy - dgtest;

        cstep(stx, fxm, dgxm, sty, fym, dgym, *stp, fm, dgm, brackt, stmin,
              stmax, infoc);

        fx = fxm + stx * dgtest;
        fy = fym + sty * dgtest;
        dgx = dgxm + dgtest;
        dgy = dgym + dgtest;
      } else {
        // This is ugly and some variables should be moved to the class scope.
        cstep(stx, fx, dgx, sty, fy, dgy, *stp, f, dg, brackt, stmin, stmax,
              infoc);
      }

      if (brackt) {
        if (fabs(sty - stx) >= 0.66 * width1) {
          *stp = stx + 0.5 * (sty - stx);
        }
        width1 = width;
        width = fabs(sty - stx);
      }
    }

    return 0;
  }

  // TODO(patwie): cpplint prefers pointers here, but this would make the code
  // unreadable. As these are all changing values a configuration structure
  // would be helpful.
  static int cstep(ScalarType &stx, ScalarType &fx, ScalarType &dx,   // NOLINT
                   ScalarType &sty,                                   // NOLINT
                   ScalarType &fy, ScalarType &dy, ScalarType &stp,   // NOLINT
                   ScalarType &fp,                                    // NOLINT
                   ScalarType &dp, bool &brackt, ScalarType &stpmin,  // NOLINT
                   ScalarType &stpmax, int &info) {                   // NOLINT
    info = 0;
    bool bound = false;

    // Check the input parameters for errors.
    if ((brackt && ((stp <= std::min<ScalarType>(stx, sty)) ||
                    (stp >= std::max<ScalarType>(stx, sty)))) ||
        (dx * (stp - stx) >= 0.0) || (stpmax < stpmin)) {
      return -1;
    }

    ScalarType sgnd = dp * (dx / fabs(dx));

    ScalarType stpf = 0;
    ScalarType stpc = 0;
    ScalarType stpq = 0;

    if (fp > fx) {
      info = 1;
      bound = true;
      ScalarType theta = 3. * (fx - fp) / (stp - stx) + dx + dp;
      ScalarType s = max_abs(theta, dx, dp);
      ScalarType gamma =
          s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp < stx) gamma = -gamma;
      ScalarType p = (gamma - dx) + theta;
      ScalarType q = ((gamma - dx) + gamma) + dp;
      ScalarType r = p / q;
      stpc = stx + r * (stp - stx);
      stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.) * (stp - stx);
      if (fabs(stpc - stx) < fabs(stpq - stx))
        stpf = stpc;
      else
        stpf = stpc + (stpq - stpc) / 2;
      brackt = true;
    } else if (sgnd < 0.0) {
      info = 2;
      bound = false;
      ScalarType theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      ScalarType s = max_abs(theta, dx, dp);
      ScalarType gamma =
          s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp > stx) gamma = -gamma;

      ScalarType p = (gamma - dp) + theta;
      ScalarType q = ((gamma - dp) + gamma) + dx;
      ScalarType r = p / q;
      stpc = stp + r * (stx - stp);
      stpq = stp + (dp / (dp - dx)) * (stx - stp);
      if (fabs(stpc - stp) > fabs(stpq - stp))
        stpf = stpc;
      else
        stpf = stpq;
      brackt = true;
    } else if (fabs(dp) < fabs(dx)) {
      info = 3;
      bound = true;
      ScalarType theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      ScalarType s = max_abs(theta, dx, dp);
      ScalarType gamma =
          s * sqrt(std::max<ScalarType>(
                  static_cast<ScalarType>(0.),
                  (theta / s) * (theta / s) - (dx / s) * (dp / s)));
      if (stp > stx) gamma = -gamma;
      ScalarType p = (gamma - dp) + theta;
      ScalarType q = (gamma + (dx - dp)) + gamma;
      ScalarType r = p / q;
      if ((r < 0.0) & (gamma != 0.0)) {
        stpc = stp + r * (stx - stp);
      } else if (stp > stx) {
        stpc = stpmax;
      } else {
        stpc = stpmin;
      }
      stpq = stp + (dp / (dp - dx)) * (stx - stp);
      if (brackt) {
        if (fabs(stp - stpc) < fabs(stp - stpq)) {
          stpf = stpc;
        } else {
          stpf = stpq;
        }
      } else {
        if (fabs(stp - stpc) > fabs(stp - stpq)) {
          stpf = stpc;
        } else {
          stpf = stpq;
        }
      }
    } else {
      info = 4;
      bound = false;
      if (brackt) {
        ScalarType theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
        ScalarType s = max_abs(theta, dy, dp);
        ScalarType gamma =
            s * sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
        if (stp > sty) gamma = -gamma;

        ScalarType p = (gamma - dp) + theta;
        ScalarType q = ((gamma - dp) + gamma) + dy;
        ScalarType r = p / q;
        stpc = stp + r * (sty - stp);
        stpf = stpc;
      } else if (stp > stx) {
        stpf = stpmax;
      } else {
        stpf = stpmin;
      }
    }

    if (fp > fx) {
      sty = stp;
      fy = fp;
      dy = dp;
    } else {
      if (sgnd < 0.0) {
        sty = stx;
        fy = fx;
        dy = dx;
      }

      stx = stp;
      fx = fp;
      dx = dp;
    }

    stpf = std::clamp(stpf, stpmin, stpmax);
    stp = stpf;

    if (brackt & bound) {
      if (sty > stx) {
        stp = std::min<ScalarType>(
            stx + static_cast<ScalarType>(0.66) * (sty - stx), stp);
      } else {
        stp = std::max<ScalarType>(
            stx + static_cast<ScalarType>(0.66) * (sty - stx), stp);
      }
    }

    return 0;
  }

  static ScalarType max_abs(ScalarType x, ScalarType y, ScalarType z) {
    return std::max(std::abs(x), std::max(std::abs(y), std::abs(z)));
  }
};
}  // namespace cppoptlib::solver::linesearch

#endif  // INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_
