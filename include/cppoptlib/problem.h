#ifndef PROBLEM_H
#define PROBLEM_H

#include <vector>
#include <Eigen/Core>

#include "meta.h"

namespace cppoptlib {

template<typename Scalar_, int Dim_ = Eigen::Dynamic>
class Problem {
 public:
  static const int Dim = Dim_;
  typedef Scalar_ Scalar;
  using TVector   = Eigen::Matrix<Scalar, Dim, 1>;
  using THessian  = Eigen::Matrix<Scalar, Dim, Dim>;
  using TCriteria = Criteria<Scalar>;

 public:
  Problem() {}
  virtual ~Problem()= default;

  virtual bool callback(const Criteria<Scalar> &state, const TVector &x) {
    return true;
  }

  /**
   * @brief returns objective value in x
   * @details [long description]
   *
   * @param x [description]
   * @return [description]
   */
  virtual Scalar value(const  TVector &x) = 0;
  /**
   * @brief overload value for nice syntax
   * @details [long description]
   *
   * @param x [description]
   * @return [description]
   */
  Scalar operator()(const  TVector &x) {
    return value(x);
  }
  /**
   * @brief returns gradient in x as reference parameter
   * @details should be overwritten by symbolic gradient
   *
   * @param grad [description]
   */
  virtual void gradient(const  TVector &x,  TVector &grad) {
    finiteGradient(x, grad);
  }

  /**
   * @brief This computes the hessian
   * @details should be overwritten by symbolic hessian, if solver relies on hessian
   */
  virtual void hessian(const TVector &x, THessian &hessian) {
    finiteHessian(x, hessian);

  }

  virtual bool checkGradient(const TVector &x, int accuracy = 3) {
    // TODO: check if derived class exists:
    // int(typeid(&Rosenbrock<double>::gradient) == typeid(&Problem<double>::gradient)) == 1 --> overwritten
    const Eigen::Index D = x.rows();
    TVector actual_grad(D);
    TVector expected_grad(D);
    gradient(x, actual_grad);
    finiteGradient(x, expected_grad, accuracy);
    for (Eigen::Index d = 0; d < D; ++d) {
      Scalar scale = std::max((std::max(fabs(actual_grad[d]), fabs(expected_grad[d]))), 1.);
      if(fabs(actual_grad[d]-expected_grad[d])>1e-2 * scale)
        return false;
    }
    return true;

  }

  virtual bool checkHessian(const TVector &x, int accuracy = 3) {
    // TODO: check if derived class exists:
    // int(typeid(&Rosenbrock<double>::gradient) == typeid(&Problem<double>::gradient)) == 1 --> overwritten
    const Eigen::Index D = x.rows();

    THessian actual_hessian = THessian::Zero(D, D);
    THessian expected_hessian = THessian::Zero(D, D);
    hessian(x, actual_hessian);
    finiteHessian(x, expected_hessian, accuracy);
    for (Eigen::Index d = 0; d < D; ++d) {
      for (Eigen::Index e = 0; e < D; ++e) {
        Scalar scale = std::max(static_cast<Scalar>(std::max(fabs(actual_hessian(d, e)), fabs(expected_hessian(d, e)))), Scalar(1.));
        if(fabs(actual_hessian(d, e)- expected_hessian(d, e))>1e-1 * scale)
          return false;
      }
    }
    return true;
  }

  virtual void finiteGradient(const  TVector &x, TVector &grad, int accuracy = 0) final {
    // accuracy can be 0, 1, 2, 3
    const Scalar eps = 2.2204e-6;
    const Scalar coeff[4][8] = {
	  {1, -1, 0, 0, 0, 0, 0, 0}, 
	  {1, -8, 8, -1, 0, 0, 0, 0},
	  {-1, 9, -45, 45, -9, 1, 0, 0},
	  {3, -32, 168, -672, 672, -168, 32, -3}
	};
    const Scalar coeff2[4][8] = { 
      {1, -1, 0, 0, 0, 0, 0, 0}, 
	  {-2, -1, 1, 2, 0, 0, 0, 0}, 
	  {-3, -2, -1, 1, 2, 3, 0, 0},
	  {-4, -3, -2, -1, 1, 2, 3, 4} 
	};
    const Scalar dd[4] = {2, 12, 60, 840};

    TVector finiteDiff(x.rows());
    TVector xx(x.rows());
	for (Eigen::Index d = 0; d < x.rows(); d++) {
      finiteDiff[d] = 0;
      for (int s = 0; s < 2*(accuracy+1); ++s)
      {
        xx = x;
        xx[d] += coeff2[accuracy][s]*eps;
        finiteDiff[d] += coeff[accuracy][s]*value(xx);
      }
      finiteDiff[d] /= (dd[accuracy]* eps);
    }
    grad = finiteDiff;
  }

  virtual void finiteHessian(const TVector &x, THessian &hessian, int accuracy = 0) final {
    const Scalar eps = std::numeric_limits<Scalar>::epsilon()*10e7;

	TVector xx(x.rows());
    if(accuracy == 0) {
      for (Eigen::Index i = 0; i < x.rows(); i++) {
        for (Eigen::Index j = 0; j < x.rows(); j++) {
          xx = x;
          Scalar f4 = value(xx);
          xx[i] += eps;
          xx[j] += eps;
          Scalar f1 = value(xx);
          xx[j] -= eps;
          Scalar f2 = value(xx);
          xx[j] += eps;
          xx[i] -= eps;
          Scalar f3 = value(xx);
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
      for (Eigen::Index i = 0; i < x.rows(); i++) {
        for (Eigen::Index j = 0; j < x.rows(); j++) {

          Scalar term_1 = 0;
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += -2*eps;  term_1 += value(xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += -1*eps;  term_1 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += 1*eps;   term_1 += value(xx);
          xx = x.eval(); xx[i] += -1*eps; xx[j] += 2*eps;   term_1 += value(xx);

          Scalar term_2 = 0;
          xx = x.eval(); xx[i] += -1*eps; xx[j] += -2*eps;  term_2 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += -1*eps;  term_2 += value(xx);
          xx = x.eval(); xx[i] += 1*eps;  xx[j] += 2*eps;   term_2 += value(xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += 1*eps;   term_2 += value(xx);

          Scalar term_3 = 0;
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += -2*eps;  term_3 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += 2*eps;   term_3 += value(xx);
          xx = x.eval(); xx[i] += -2*eps; xx[j] += -2*eps;  term_3 -= value(xx);
          xx = x.eval(); xx[i] += 2*eps;  xx[j] += 2*eps;   term_3 -= value(xx);

          Scalar term_4 = 0;
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
