// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
//
#ifndef INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_
#define INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_

#include <cmath>

namespace cppoptlib {
namespace solver {
namespace linesearch {

template <typename Function, int Ord>
class MoreThuente {
 public:
  using ScalarT = typename Function::ScalarT;
  using VectorT = typename Function::VectorT;

  /**
   * @brief use MoreThuente Rule for (strong) Wolfe conditiions
   * @details [long description]
   *
   * @param searchDir search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */

  static ScalarT search(const VectorT &x, const VectorT &searchDir,
                       const Function &function,
                       const ScalarT alpha_init = 1.0) {
    // Assumed step width.
    ScalarT ak = alpha_init;

    ScalarT fval = function(x);
    VectorT g = x.eval();
    function.Gradient(x, &g);

    VectorT s = searchDir.eval();
    VectorT xx = x.eval();

    cvsrch(function, xx, fval, g, ak, s);

    return ak;
  }

  static int cvsrch(const Function &function, VectorT &x, ScalarT f, VectorT &g,
                    ScalarT &stp, VectorT &s) {
    // we rewrite this from MIN-LAPACK and some MATLAB code
    int info = 0;
    int infoc = 1;
    const ScalarT xtol = 1e-15;
    const ScalarT ftol = 1e-4;
    const ScalarT gtol = 1e-2;
    const ScalarT stpmin = 1e-15;
    const ScalarT stpmax = 1e15;
    const ScalarT xtrapf = 4;
    const int maxfev = 20;
    int nfev = 0;

    ScalarT dginit = g.dot(s);
    if (dginit >= 0.0) {
      // no descent direction
      // TODO: handle this case
      return -1;
    }

    bool brackt = false;
    bool stage1 = true;

    ScalarT finit = f;
    ScalarT dgtest = ftol * dginit;
    ScalarT width = stpmax - stpmin;
    ScalarT width1 = 2 * width;
    VectorT wa = x.eval();

    ScalarT stx = 0.0;
    ScalarT fx = finit;
    ScalarT dgx = dginit;
    ScalarT sty = 0.0;
    ScalarT fy = finit;
    ScalarT dgy = dginit;

    ScalarT stmin;
    ScalarT stmax;

    while (true) {
      // make sure we stay in the interval when setting min/max-step-width
      if (brackt) {
        stmin = std::min<ScalarT>(stx, sty);
        stmax = std::max<ScalarT>(stx, sty);
      } else {
        stmin = stx;
        stmax = stp + xtrapf * (stp - stx);
      }

      // Force the step to be within the bounds stpmax and stpmin.
      stp = std::max<ScalarT>(stp, stpmin);
      stp = std::min<ScalarT>(stp, stpmax);

      // Oops, let us return the last reliable values
      if ((brackt && ((stp <= stmin) || (stp >= stmax))) ||
          (nfev >= maxfev - 1) || (infoc == 0) ||
          (brackt && ((stmax - stmin) <= (xtol * stmax)))) {
        stp = stx;
      }

      // test new point
      x = wa + stp * s;
      f = function(x);
      function.Gradient(x, &g);
      nfev++;
      ScalarT dg = g.dot(s);
      ScalarT ftest1 = finit + stp * dgtest;

      // all possible convergence tests
      if ((brackt & ((stp <= stmin) | (stp >= stmax))) | (infoc == 0)) info = 6;

      if ((stp == stpmax) & (f <= ftest1) & (dg <= dgtest)) info = 5;

      if ((stp == stpmin) & ((f > ftest1) | (dg >= dgtest))) info = 4;

      if (nfev >= maxfev) info = 3;

      if (brackt & (stmax - stmin <= xtol * stmax)) info = 2;

      if ((f <= ftest1) & (fabs(dg) <= gtol * (-dginit))) info = 1;

      // terminate when convergence reached
      if (info != 0) return -1;

      if (stage1 & (f <= ftest1) &
          (dg >= std::min<ScalarT>(ftol, gtol) * dginit))
        stage1 = false;

      if (stage1 & (f <= fx) & (f > ftest1)) {
        ScalarT fm = f - stp * dgtest;
        ScalarT fxm = fx - stx * dgtest;
        ScalarT fym = fy - sty * dgtest;
        ScalarT dgm = dg - dgtest;
        ScalarT dgxm = dgx - dgtest;
        ScalarT dgym = dgy - dgtest;

        cstep(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin,
              stmax, infoc);

        fx = fxm + stx * dgtest;
        fy = fym + sty * dgtest;
        dgx = dgxm + dgtest;
        dgy = dgym + dgtest;
      } else {
        // this is ugly and some variables should be moved to the class scope
        cstep(stx, fx, dgx, sty, fy, dgy, stp, f, dg, brackt, stmin, stmax,
              infoc);
      }

      if (brackt) {
        if (fabs(sty - stx) >= 0.66 * width1) stp = stx + 0.5 * (sty - stx);
        width1 = width;
        width = fabs(sty - stx);
      }
    }

    return 0;
  }

  static int cstep(ScalarT &stx, ScalarT &fx, ScalarT &dx, ScalarT &sty, ScalarT &fy,
                   ScalarT &dy, ScalarT &stp, ScalarT &fp, ScalarT &dp,
                   bool &brackt, ScalarT &stpmin, ScalarT &stpmax, int &info) {
    info = 0;
    bool bound = false;

    // Check the input parameters for errors.
    if ((brackt & ((stp <= std::min<ScalarT>(stx, sty)) |
                   (stp >= std::max<ScalarT>(stx, sty)))) |
        (dx * (stp - stx) >= 0.0) | (stpmax < stpmin)) {
      return -1;
    }

    ScalarT sgnd = dp * (dx / fabs(dx));

    ScalarT stpf = 0;
    ScalarT stpc = 0;
    ScalarT stpq = 0;

    if (fp > fx) {
      info = 1;
      bound = true;
      ScalarT theta = 3. * (fx - fp) / (stp - stx) + dx + dp;
      ScalarT s = std::max<ScalarT>(theta, std::max<ScalarT>(dx, dp));
      ScalarT gamma = s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp < stx) gamma = -gamma;
      ScalarT p = (gamma - dx) + theta;
      ScalarT q = ((gamma - dx) + gamma) + dp;
      ScalarT r = p / q;
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
      ScalarT theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      ScalarT s = std::max<ScalarT>(theta, std::max<ScalarT>(dx, dp));
      ScalarT gamma = s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp > stx) gamma = -gamma;

      ScalarT p = (gamma - dp) + theta;
      ScalarT q = ((gamma - dp) + gamma) + dx;
      ScalarT r = p / q;
      stpc = stp + r * (stx - stp);
      stpq = stp + (dp / (dp - dx)) * (stx - stp);
      if (fabs(stpc - stp) > fabs(stpq - stp))
        stpf = stpc;
      else
        stpf = stpq;
      brackt = true;
    } else if (fabs(dp) < fabs(dx)) {
      info = 3;
      bound = 1;
      ScalarT theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      ScalarT s = std::max<ScalarT>(theta, std::max<ScalarT>(dx, dp));
      ScalarT gamma = s * sqrt(std::max<ScalarT>(
                             static_cast<ScalarT>(0.),
                             (theta / s) * (theta / s) - (dx / s) * (dp / s)));
      if (stp > stx) gamma = -gamma;
      ScalarT p = (gamma - dp) + theta;
      ScalarT q = (gamma + (dx - dp)) + gamma;
      ScalarT r = p / q;
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
        ScalarT theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
        ScalarT s = std::max<ScalarT>(theta, std::max<ScalarT>(dy, dp));
        ScalarT gamma =
            s * sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
        if (stp > sty) gamma = -gamma;

        ScalarT p = (gamma - dp) + theta;
        ScalarT q = ((gamma - dp) + gamma) + dy;
        ScalarT r = p / q;
        stpc = stp + r * (sty - stp);
        stpf = stpc;
      } else if (stp > stx)
        stpf = stpmax;
      else {
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

    stpf = std::min<ScalarT>(stpmax, stpf);
    stpf = std::max<ScalarT>(stpmin, stpf);
    stp = stpf;

    if (brackt & bound) {
      if (sty > stx) {
        stp = std::min<ScalarT>(stx + static_cast<ScalarT>(0.66) * (sty - stx),
                               stp);
      } else {
        stp = std::max<ScalarT>(stx + static_cast<ScalarT>(0.66) * (sty - stx),
                               stp);
      }
    }

    return 0;
  }
};
};  // namespace linesearch
};  // namespace solver
}  // namespace cppoptlib

#endif  // INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_