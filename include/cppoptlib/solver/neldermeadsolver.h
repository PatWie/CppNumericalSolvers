// CppNumericalSolver
#ifndef NELDERMEADSOLVER_H_
#define NELDERMEADSOLVER_H_
#include <cmath>
#include <Eigen/Dense>
#include "isolver.h"

namespace cppoptlib {

template<typename T>
class NelderMeadSolver : public ISolver<T, 0> {
  using ISolver<T, 0>::ISolver; // Inherit the constructors from the interface
 public:
  /**
   * @brief minimize
   * @details [long description]
   *
   * @param objFunc [description]
   */
  void minimize(Problem<T> &objFunc, Vector<T> & x) {

    const T rho = 1.;    // rho > 0
    const T xi  = 2.;    // xi  > max(rho, 1)
    const T gam = 0.5;   // 0 < gam < 1

    const size_t DIM = x.rows();

    // create initial simplex
    Matrix<T> x0 = Matrix<T>::Zero(DIM, DIM + 1);
    for (int c = 0; c < DIM + 1; ++c) {
      for (int r = 0; r < DIM; ++r) {
        x0(r, c) = x(r);
        if (r == c - 1) {
          if (x(r) == 0) {
            x0(r, c) = 0.00025;
          } else {

          }
          x0(r, c) = (1 + 0.05) * x(r);
        }
      }
    }

    // compute function values
    std::vector<T> f; f.resize(DIM + 1);
    std::vector<int> index; index.resize(DIM + 1);
    for (int i = 0; i < DIM + 1; ++i) {
      f[i] = objFunc(static_cast<Vector<T> >(x0.col(i)));
      index[i] = i;
    }

    sort(index.begin(), index.end(), [&](int a, int b)-> bool { return f[a] < f[b]; });

    const int maxIter = this->m_ctrl.iterations*DIM;
    this->m_info.iterations = 0;
    while (this->m_info.iterations < maxIter) {

      // conv-check
      T max1 = fabs(f[index[1]] - f[index[0]]);
      T max2 = (x0.col(index[1]) - x0.col(index[0]) ).array().abs().maxCoeff();
      for (int i = 2; i < DIM + 1; ++i) {
        T tmp1 = fabs(f[index[i]] - f[index[0]]);
        if (tmp1 > max1)
          max1 = tmp1;

        T tmp2 = (x0.col(index[i]) - x0.col(index[0]) ).array().abs().maxCoeff();
        if (tmp2 > max2)
          max2 = tmp2;
      }
      const T tt1 = std::max(static_cast<T>(1.e-04), 10 * std::nextafter(f[index[0]], std::numeric_limits<T>::epsilon()) - f[index[0]]);
      const T tt2 = std::max(static_cast<T>(1.e-04), 10 * (std::nextafter(x0.col(index[0]).maxCoeff(), std::numeric_limits<T>::epsilon())
                    - x0.col(index[0]).maxCoeff()));

      // max(||x - shift(x) ||_inf ) <= tol,
      if (max1 <=  tt1) {
        // values to similar
        if (max2 <= tt2) {
          break;
        }
      }

      //////////////////////////

      // midpoint of the simplex opposite the worst point
      Vector<T> x_bar = Vector<T>::Zero(DIM);
      for (int i = 0; i < DIM; ++i) {
        x_bar += x0.col(index[i]);
      }
      x_bar /= (T)DIM;

      // Compute the reflection point
      const Vector<T> x_r   = ( 1. + rho ) * x_bar - rho   * x0.col(index[DIM]);
      const T f_r = objFunc(x_r);

      if (f_r < f[index[0]]) {
        // the expansion point
        const Vector<T> x_e = ( 1. + rho * xi ) * x_bar - rho * xi   * x0.col(index[DIM]);
        const T f_e = objFunc(x_e);
        if ( f_e < f_r ) {
          // expand
          x0.col(index[DIM]) = x_e;
          f[index[DIM]] = f_e;
        } else {
          // reflect
          x0.col(index[DIM]) = x_r;
          f[index[DIM]] = f_r;
        }
      } else {
        if ( f_r < f[index[DIM]] ) {
          x0.col(index[DIM]) = x_r;
          f[index[DIM]] = f_r;
        } else {
          // contraction
          if (f_r < f[index[DIM]]) {
            const Vector<T> x_c = (1 + rho * gam) * x_bar - rho * gam * x0.col(index[DIM]);
            const T f_c = objFunc(x_c);
            if ( f_c <= f_r ) {
              // outside
              x0.col(index[DIM]) = x_c;
              f[index[DIM]] = f_c;
            } else {
              shrink(x0, index, f, objFunc);
            }
          } else {
            // inside
            const Vector<T> x_c = ( 1 - gam ) * x_bar + gam   * x0.col(index[DIM]);
            const T f_c = objFunc(x_c);
            if (f_c < f[index[DIM]]) {
              x0.col(index[DIM]) = x_c;
              f[index[DIM]] = f_c;
            } else {
              shrink(x0, index, f, objFunc);
            }
          }
        }
      }
      sort(index.begin(), index.end(), [&](int a, int b)-> bool { return f[a] < f[b]; });
      ++this->m_info.iterations;
    }
    x = x0.col(index[0]);
  }

  void shrink(Matrix<T> &x, std::vector<int> &index, std::vector<T> &f, Problem<T> &objFunc) {

    const T sig = 0.5;   // 0 < sig < 1
    const int DIM = x.rows();
    f[index[0]] = objFunc(x.col(index[0]));
    for (int i = 1; i < DIM + 1; ++i) {
      x.col(index[i]) = sig * x.col(index[i]) + (1. - sig) * x.col(index[0]);
      f[index[i]] = objFunc(x.col(index[i]));
    }

  }

};

} /* namespace cppoptlib */

#endif /* NELDERMEADSOLVER_H_ */
