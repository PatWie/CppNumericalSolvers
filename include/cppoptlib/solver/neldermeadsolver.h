// CppNumericalSolver
#ifndef NELDERMEADSOLVER_H_
#define NELDERMEADSOLVER_H_
#include <cmath>
#include <Eigen/Core>
#include "isolver.h"

namespace cppoptlib {

template<typename ProblemType>
class NelderMeadSolver : public ISolver<ProblemType, 0> {
 public:
  using Superclass = ISolver<ProblemType, 0>;
  using typename Superclass::Scalar;
  using typename Superclass::TVector;
  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  /**
   * @brief minimize
   * @details [long description]
   *
   * @param objFunc [description]
   */
  void minimize(ProblemType &objFunc, TVector & x) {

    const Scalar rho = 1.;    // rho > 0
    const Scalar xi  = 2.;    // xi  > max(rho, 1)
    const Scalar gam = 0.5;   // 0 < gam < 1

    const size_t DIM = x.rows();

    // create initial simplex
    MatrixType x0 = MatrixType::Zero(DIM, DIM + 1);
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
        if (x0(r, c) > objFunc.upperBound()[r])
          x0(r, c) = objFunc.upperBound()[r];
        else if (x0(r, c) < objFunc.lowerBound()[r])
          x0(r, c) = objFunc.lowerBound()[r];
      }
    }

    // compute function values
    std::vector<Scalar> f; f.resize(DIM + 1);
    std::vector<int> index; index.resize(DIM + 1);
    for (int i = 0; i < DIM + 1; ++i) {
      f[i] = objFunc(static_cast<TVector >(x0.col(i)));
      index[i] = i;
    }

    sort(index.begin(), index.end(), [&](int a, int b)-> bool { return f[a] < f[b]; });

    int iter = 0;
    const int maxIter = this->m_stop.iterations*DIM;
    while (objFunc.callback(this->m_current, x0.col(index[0])) && (iter < maxIter)) {

      // conv-check
      Scalar max1 = fabs(f[index[1]] - f[index[0]]);
      Scalar max2 = (x0.col(index[1]) - x0.col(index[0]) ).array().abs().maxCoeff();
      for (int i = 2; i < DIM + 1; ++i) {
        Scalar tmp1 = fabs(f[index[i]] - f[index[0]]);
        if (tmp1 > max1)
          max1 = tmp1;

        Scalar tmp2 = (x0.col(index[i]) - x0.col(index[0]) ).array().abs().maxCoeff();
        if (tmp2 > max2)
          max2 = tmp2;
      }
      const Scalar tt1 = std::max(Scalar(1.e-04), 10 * std::nextafter(f[index[0]], std::numeric_limits<Scalar>::epsilon()) - f[index[0]]);
      const Scalar tt2 = std::max(Scalar(1.e-04), 10 * (std::nextafter(x0.col(index[0]).maxCoeff(), std::numeric_limits<Scalar>::epsilon())
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
      TVector x_bar = TVector::Zero(DIM);
      for (int i = 0; i < DIM; ++i) {
        x_bar += x0.col(index[i]);
      }
      x_bar /= Scalar(DIM);

      // Compute the reflection point
      const TVector x_r   = clamp_x(objFunc, ( 1. + rho ) * x_bar - rho   * x0.col(index[DIM]));
      const Scalar f_r = objFunc(x_r);

      if (f_r < f[index[0]]) {
        // the expansion point
        const TVector x_e = clamp_x(objFunc, ( 1. + rho * xi ) * x_bar - rho * xi   * x0.col(index[DIM]));
        const Scalar f_e = objFunc(x_e);
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
            const TVector x_c = clamp_x(objFunc, (1 + rho * gam) * x_bar - rho * gam * x0.col(index[DIM]));
            const Scalar f_c = objFunc(x_c);
            if ( f_c <= f_r ) {
              // outside
              x0.col(index[DIM]) = x_c;
              f[index[DIM]] = f_c;
            } else {
              shrink(x0, index, f, objFunc);
            }
          } else {
            // inside
            const TVector x_c = clamp_x(objFunc, ( 1 - gam ) * x_bar + gam   * x0.col(index[DIM]));
            const Scalar f_c = objFunc(x_c);
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
      iter++;
    }
    x = x0.col(index[0]);
  }
  
  TVector clamp_x(ProblemType &objFunc, const TVector &x) {
      TVector xx = TVector::Zero(x.rows());
      for (int i = 0; i < x.rows(); i++) {
          if (x[i] < objFunc.lowerBound()[i])
              xx[i] = objFunc.lowerBound()[i];
          else if (x[i] > objFunc.upperBound()[i])
              xx[i] = objFunc.upperBound()[i];
          else
              xx[i] = x[i];
      }
      return xx;
  }

  void shrink(MatrixType &x, std::vector<int> &index, std::vector<Scalar> &f, ProblemType &objFunc) {

    const Scalar sig = 0.5;   // 0 < sig < 1
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
