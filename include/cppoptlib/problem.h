#ifndef PROBLEM_H
#define PROBLEM_H

#include <Eigen/Dense>
#ifndef MATLAB
#include "../gtest/gtest.h"
#else
#define EXPECT_NEAR(x,y,z)
#endif

#include "meta.h"

namespace cppoptlib {
template<typename T>
class Problem {
 protected:

 public:

  bool hasLowerBound = false;
  bool hasUpperBound = false;

  Vector<T> lowerBound;
  Vector<T> upperBound;

  Problem() {}

  /**
   * @brief returns objective value in x
   * @details [long description]
   *
   * @param x [description]
   * @return [description]
   */
  virtual T value(const  Vector<T> &x) = 0;
  /**
   * @brief overload value for nice syntax
   * @details [long description]
   *
   * @param x [description]
   * @return [description]
   */
  T operator()(const  Vector<T> &x) {
    return value(x);
  }
  /**
   * @brief returns gradient in x as reference parameter
   * @details should be overwritten by symbolic gradient
   *
   * @param grad [description]
   */
  virtual void gradient(const  Vector<T> &x,  Vector<T> &grad) {
    finiteGradient(x, grad);
  }

  /**
   * @brief This computes the hessian
   * @details should be overwritten by symbolic hessian, if solver relies on hessian
   */
  virtual void hessian(const Vector<T> & x, Matrix<T> & hessian) {
    finiteHessian(x, hessian);

  }

  virtual bool checkGradient(const Vector<T> & x, int accuracy = 3) {
    // TODO: check if derived class exists:
    // int(typeid(&Rosenbrock<double>::gradient) == typeid(&Problem<double>::gradient)) == 1 --> overwritten
    const int D = x.rows();
    Vector<T> actual_grad(D);
    Vector<T> expected_grad(D);
    gradient(x, actual_grad);
    finiteGradient(x, expected_grad, accuracy);

    bool correct = true;

    for (int d = 0; d < D; ++d) {
      T scale = std::max(std::max(fabs(actual_grad[d]), fabs(expected_grad[d])), 1.);
      EXPECT_NEAR(actual_grad[d], expected_grad[d], 1e-2 * scale);
      if(fabs(actual_grad[d]-expected_grad[d])>1e-2 * scale)
        correct = false;
    }
    return correct;

  }

  virtual bool checkHessian(const Vector<T> & x, int accuracy = 3) {
    // TODO: check if derived class exists:
    // int(typeid(&Rosenbrock<double>::gradient) == typeid(&Problem<double>::gradient)) == 1 --> overwritten
    const int D = x.rows();
    bool correct = true;

    Matrix<double> actual_hessian = Matrix<double>::Zero(D, D);
    Matrix<double> expected_hessian = Matrix<double>::Zero(D, D);
    hessian(x, actual_hessian);
    finiteHessian(x, expected_hessian, accuracy);
    for (int d = 0; d < D; ++d) {
      for (int e = 0; e < D; ++e) {
        T scale = std::max(std::max(fabs(actual_hessian(d, e)), fabs(expected_hessian(d, e))), 1.);
        EXPECT_NEAR(actual_hessian(d, e), expected_hessian(d, e), 1e-1 * scale);
        if(fabs(actual_hessian(d, e)- expected_hessian(d, e))>1e-1 * scale)
        correct = false;
      }
    }
    return correct;

  }

  virtual void finiteGradient(const  Vector<T> &x, Vector<T> &grad, int accuracy = 0) final {
    // accuracy can be 0, 1, 2, 3
    const T eps = 2.2204e-8;
    const size_t D = x.rows();
    const int idx = (accuracy-3)/2;
    const std::vector< std::vector <T>> coeff =
    { {1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3} };
    const std::vector< std::vector <T>> coeff2 =
    { {1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4} };
    const std::vector <T> dd = {2, 12, 60, 840};

    Vector<T> finiteDiff(D);
    for (size_t d = 0; d < D; d++) {
      finiteDiff[d] = 0;
      for (int s = 0; s < 2*(accuracy+1); ++s)
      {
        Vector<T> xx = x.eval();
        xx[d] += coeff2[accuracy][s]*eps;
        finiteDiff[d] += coeff[accuracy][s]*value(xx);
      }
      finiteDiff[d] /= (dd[accuracy]* eps);
    }
    grad = finiteDiff;
  }

  virtual void finiteHessian(const Vector<T> & x, Matrix<T> & hessian, int accuracy = 0) final {
    const T eps = 2.2204e-08;
    const size_t DIM = x.rows();

    if(accuracy == 0) {
      for (size_t i = 0; i < DIM; i++) {
        for (size_t j = 0; j < DIM; j++) {
          Vector<T> xx = x;
          T f4 = value(xx);
          xx[i] += eps;
          xx[j] += eps;
          T f1 = value(xx);
          xx[j] -= eps;
          T f2 = value(xx);
          xx[j] += eps;
          xx[i] -= eps;
          T f3 = value(xx);
          hessian(i, j) = (f1 - f2 - f3 + f4) / (eps * eps);
        }
      }
    } else {
      /*
        \displaystyle{{\frac{\partial^2{f}}{\partial{x}\partial{y}}}\approx
        \frac{1}{600\,h^2} \left[\begin{matrix}
          -63(f_{1,-2}+f_{2,-1}+f_{-2,1}+f_{-1,2})+\\
          63(f_{-1,-2}+f_{-2,-1}+f_{1,2}+f_{2,1})+\\
          44(f_{2,-2}+f_{-2,2}-f_{-2,-2}-f_{2,2})+\\
          74(f_{-1,-1}+f_{1,1}-f_{1,-1}-f_{-1,1})
        \end{matrix}\right] }
      */
      Vector<T> xx;
      for (size_t i = 0; i < DIM; i++) {
        for (size_t j = 0; j < DIM; j++) {

          T term_1 = 0;
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += -2*eps;  term_1 += value(xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += -1*eps;  term_1 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += 1*eps;   term_1 += value(xx);
          xx = x.eval(); xx[i] += -1*eps; xx[j] += 2*eps;   term_1 += value(xx);

          T term_2 = 0;
          xx = x.eval(); xx[i] += -1*eps; xx[j] += -2*eps;  term_2 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += -1*eps;  term_2 += value(xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += 2*eps;   term_2 += value(xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += 1*eps;   term_2 += value(xx);

          T term_3 = 0;
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += -2*eps;  term_3 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += 2*eps;   term_3 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += -2*eps;  term_3 -= value(xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += 2*eps;   term_3 -= value(xx);

          T term_4 = 0;
          xx = x.eval(); xx[i] += -1*eps; xx[j] += -1*eps;  term_4 += value(xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += 1*eps;   term_4 += value(xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += -1*eps;  term_4 -= value(xx);
          xx = x.eval(); xx[i] += -1*eps; xx[j] += 1*eps;   term_4 -= value(xx);

          hessian(i, j) = (-63 * term_1+63 * term_2+44 * term_3+74 * term_4)/(600.0 * eps * eps);

        }
      }
    }

  }

};
}

#endif /* PROBLEM_H */