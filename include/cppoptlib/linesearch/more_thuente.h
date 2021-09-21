// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
//
#ifndef INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_
#define INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_

#include <algorithm>
#include <cmath>

namespace cppoptlib::solver::linesearch {

template <typename Function, int Ord>
class MoreThuente {
 public:
  using scalar_t = typename Function::scalar_t;
  using vector_t = typename Function::vector_t;
  using state_t = typename Function::state_t;

  /**
   * @brief use MoreThuente Rule for (strong) Wolfe conditiions
   * @details [long description]
   *
   * @param search_direction search direction for next update step
   * @param function handle to problem
   *
   * @return step-width
   */

  static scalar_t Search(const state_t &state, const vector_t &search_direction,
                         const Function &function,
                         const scalar_t alpha_init = 1.0) {
    scalar_t alpha = alpha_init;
    vector_t g = state.gradient;

    vector_t s = search_direction.eval();
    vector_t xx = state.x;

    cvsrch(function, &xx, state.value, &g, &alpha, s);

    return alpha;
  }

  static int cvsrch(const Function &function, vector_t *x, scalar_t f,
                    vector_t *g, scalar_t *stp, const vector_t &s) {
    // we rewrite this from MIN-LAPACK and some MATLAB code
    int info = 0;
    int infoc = 1;
    constexpr scalar_t xtol = 1e-15;
    constexpr scalar_t ftol = 1e-4;
    constexpr scalar_t gtol = 1e-2;
    constexpr scalar_t stpmin = 1e-15;
    constexpr scalar_t stpmax = 1e15;
    constexpr scalar_t xtrapf = 4;
    constexpr int maxfev = 20;
    int nfev = 0;

    scalar_t dginit = g->dot(s);
    if (dginit >= 0.0) {
      // There is no descent direction.
      // TODO(patwie): Handle this case.
      return -1;
    }

    bool brackt = false;
    bool stage1 = true;

    scalar_t finit = f;
    scalar_t dgtest = ftol * dginit;
    scalar_t width = stpmax - stpmin;
    scalar_t width1 = 2 * width;
    vector_t wa = x->eval();

    scalar_t stx = 0.0;
    scalar_t fx = finit;
    scalar_t dgx = dginit;
    scalar_t sty = 0.0;
    scalar_t fy = finit;
    scalar_t dgy = dginit;

    scalar_t stmin;
    scalar_t stmax;

    while (true) {
      // Make sure we stay in the interval when setting min/max-step-width.
      if (brackt) {
        stmin = std::min<scalar_t>(stx, sty);
        stmax = std::max<scalar_t>(stx, sty);
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
      f = function(*x);
      function.Gradient(*x, g);
      nfev++;
      scalar_t dg = g->dot(s);
      scalar_t ftest1 = finit + *stp * dgtest;

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
          (dg >= std::min<scalar_t>(ftol, gtol) * dginit))
        stage1 = false;

      if (stage1 & (f <= fx) & (f > ftest1)) {
        scalar_t fm = f - *stp * dgtest;
        scalar_t fxm = fx - stx * dgtest;
        scalar_t fym = fy - sty * dgtest;
        scalar_t dgm = dg - dgtest;
        scalar_t dgxm = dgx - dgtest;
        scalar_t dgym = dgy - dgtest;

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
  static int cstep(scalar_t &stx, scalar_t &fx, scalar_t &dx,     // NOLINT
                   scalar_t &sty,                                 // NOLINT
                   scalar_t &fy, scalar_t &dy, scalar_t &stp,     // NOLINT
                   scalar_t &fp,                                  // NOLINT
                   scalar_t &dp, bool &brackt, scalar_t &stpmin,  // NOLINT
                   scalar_t &stpmax, int &info) {                 // NOLINT
    info = 0;
    bool bound = false;

    // Check the input parameters for errors.
    if ((brackt & ((stp <= std::min<scalar_t>(stx, sty)) |
                   (stp >= std::max<scalar_t>(stx, sty)))) |
        (dx * (stp - stx) >= 0.0) | (stpmax < stpmin)) {
      return -1;
    }

    scalar_t sgnd = dp * (dx / fabs(dx));

    scalar_t stpf = 0;
    scalar_t stpc = 0;
    scalar_t stpq = 0;

    if (fp > fx) {
      info = 1;
      bound = true;
      scalar_t theta = 3. * (fx - fp) / (stp - stx) + dx + dp;
      scalar_t s = max_abs(theta, dx, dp);
      scalar_t gamma =
          s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp < stx) gamma = -gamma;
      scalar_t p = (gamma - dx) + theta;
      scalar_t q = ((gamma - dx) + gamma) + dp;
      scalar_t r = p / q;
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
      scalar_t theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      scalar_t s = max_abs(theta, dx, dp);
      scalar_t gamma =
          s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp > stx) gamma = -gamma;

      scalar_t p = (gamma - dp) + theta;
      scalar_t q = ((gamma - dp) + gamma) + dx;
      scalar_t r = p / q;
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
      scalar_t theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      scalar_t s = max_abs(theta, dx, dp);
      scalar_t gamma = s * sqrt(std::max<scalar_t>(static_cast<scalar_t>(0.),
                                                   (theta / s) * (theta / s) -
                                                       (dx / s) * (dp / s)));
      if (stp > stx) gamma = -gamma;
      scalar_t p = (gamma - dp) + theta;
      scalar_t q = (gamma + (dx - dp)) + gamma;
      scalar_t r = p / q;
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
        scalar_t theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
        scalar_t s = max_abs(theta, dy, dp);
        scalar_t gamma =
            s * sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
        if (stp > sty) gamma = -gamma;

        scalar_t p = (gamma - dp) + theta;
        scalar_t q = ((gamma - dp) + gamma) + dy;
        scalar_t r = p / q;
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
        stp = std::min<scalar_t>(
            stx + static_cast<scalar_t>(0.66) * (sty - stx), stp);
      } else {
        stp = std::max<scalar_t>(
            stx + static_cast<scalar_t>(0.66) * (sty - stx), stp);
      }
    }

    return 0;
  }

  static scalar_t max_abs(scalar_t x, scalar_t y, scalar_t z) {
    return std::max(std::abs(x), std::max(std::abs(y), std::abs(z)));
  }
};
}  // namespace cppoptlib::solver::linesearch

#endif  // INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_
