// CppNumericalSolver
#include <iostream>
#include <list>
#include <Eigen/LU>
#include "isolver.h"
#include "../linesearch/morethuente.h"

#ifndef LBFGSBSOLVER_H_
#define LBFGSBSOLVER_H_

namespace cppoptlib {

template<typename Dtype>
class LbfgsbSolver : public ISolver<Dtype, 1> {

  // last updates
  std::list<Vector<Dtype>> xHistory;
  // workspace matrices
  Matrix<Dtype> W, M;
  // ref to problem statement
  Problem<Dtype> *objFunc_;

  Dtype theta;

  int DIM;

  /**
   * @brief sort pairs (k,v) according v ascending
   * @details [long description]
   *
   * @param v [description]
   * @return [description]
   */
  std::vector<int> sort_indexes(const std::vector< std::pair<int, Dtype> > &v) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
      idx[i] = v[i].first;
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
      return v[i1].second < v[i2].second;
    });
    return idx;
  }

  /**
   * @brief Algorithm CP: Computation of the generalized Cauchy point
   * @details PAGE 8
   *
   * @param c [description]
   */
  void GetGeneralizedCauchyPoint(Vector<Dtype> &x, Vector<Dtype> &g, Vector<Dtype> &x_cauchy,
  Vector<Dtype> &c) {
    const int DIM = x.rows();
    // Given x,l,u,g, and B = \theta I-WMW

    // {all t_i} = { (idx,value), ... }
    // TODO: use "std::set" ?
    std::vector<std::pair<int, Dtype> > SetOfT;

    // the feasible set is implicitly given by "SetOfT - {t_i==0}"
    Vector<Dtype> d = Vector<Dtype>::Zero(DIM, 1);

    // n operations
    for (int j = 0; j < DIM; j++) {
      if (g(j) == 0) {
        SetOfT.push_back(std::make_pair(j, std::numeric_limits<Dtype>::max()));
      } else {
        Dtype tmp = 0;
        if (g(j) < 0) {
          tmp = (x(j) - objFunc_->upperBound(j)) / g(j);
        } else {
          tmp = (x(j) - objFunc_->lowerBound(j)) / g(j);
        }
        d(j) = -g(j);
        SetOfT.push_back(std::make_pair(j, tmp));
      }

    }
    // sortedindices [1,0,2] means the minimal element is on the 1-st entry
    std::vector<int> sortedIndices = sort_indexes(SetOfT);

    x_cauchy = x;
    // Initialize
    // p :=     W^Dtype*p
    Vector<Dtype> p = (W.transpose() * d);                     // (2mn operations)
    // c :=     0
    c = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>::Zero(M.rows(), 1);
    // f' :=    g^Dtype*d = -d^Td
    Dtype f_prime = -d.dot(d);                         // (n operations)
    // f'' :=   \theta*d^Dtype*d-d^Dtype*W*M*W^Dtype*d = -\theta*f' - p^Dtype*M*p
    Dtype f_doubleprime = (Dtype)(-1.0 * theta) * f_prime - p.dot(M * p); // (O(m^2) operations)
    // \delta t_min :=  -f'/f''
    Dtype dt_min = -f_prime / f_doubleprime;
    // t_old :=     0
    Dtype t_old = 0;
    // b :=     argmin {t_i , t_i >0}
    int i = 0;
    for (int j = 0; j < DIM; j++) {
      i = j;
      if (SetOfT[sortedIndices[j]].second != 0)
        break;
    }
    int b = sortedIndices[i];
    // see below
    // t                    :=  min{t_i : i in F}
    Dtype t = SetOfT[b].second;
    // \delta Dtype             :=  t - 0
    Dtype dt = t - t_old;

    // examination of subsequent segments
    while ((dt_min >= dt) && (i < DIM)) {
      if (d(b) > 0)
        x_cauchy(b) = objFunc_->upperBound(b);
      else if (d(b) < 0)
        x_cauchy(b) = objFunc_->lowerBound(b);

      // z_b = x_p^{cp} - x_b
      Dtype zb = x_cauchy(b) - x(b);
      // c   :=  c +\delta t*p
      c += dt * p;
      // cache
      Vector<Dtype> wbt = W.row(b);

      f_prime += dt * f_doubleprime + (Dtype) g(b) * g(b) + (Dtype) theta * g(b) * zb - (Dtype) g(b) *
      wbt.transpose() * (M * c);
      f_doubleprime += (Dtype) - 1.0 * theta * g(b) * g(b)
                       - (Dtype) 2.0 * (g(b) * (wbt.dot(M * p)))
                       - (Dtype) g(b) * g(b) * wbt.transpose() * (M * wbt);
      p += g(b) * wbt.transpose();
      d(b) = 0;
      dt_min = -f_prime / f_doubleprime;
      t_old = t;
      ++i;
      if (i < DIM) {
        b = sortedIndices[i];
        t = SetOfT[b].second;
        dt = t - t_old;
      }

    }

    dt_min = std::max(dt_min, (Dtype)0.0);
    t_old += dt_min;

    // Debug(sortedIndices[0] << " " << sortedIndices[1]);

    #pragma omp parallel for
    for (int ii = i; ii < x_cauchy.rows(); ii++) {
      x_cauchy(sortedIndices[ii]) = x(sortedIndices[ii])
                                    + t_old * d(sortedIndices[ii]);
    }
    // Debug(x_cauchy.transpose());

    c += dt_min * p;
    // Debug(c.transpose());

  }

  Dtype findAlpha(Vector<Dtype> &x_cp, Vector<Dtype> &du, std::vector<int> &FreeVariables) {
    /* this returns
     * a* = max {a : a <= 1 and  l_i-xc_i <= a*d_i <= u_i-xc_i}
     */
    Dtype alphastar = 1;
    const unsigned int n = FreeVariables.size();
    for (unsigned int i = 0; i < n; i++) {
      if (du(i) > 0) {
        alphastar = std::min(alphastar, (objFunc_->upperBound(FreeVariables[i]) - x_cp(FreeVariables[i])) / du(i));
      } else {
        alphastar = std::min(alphastar, (objFunc_->lowerBound(FreeVariables[i]) - x_cp(FreeVariables[i])) / du(i));
      }
    }
    return alphastar;
  }

  void SubspaceMinimization(Vector<Dtype> &x_cauchy, Vector<Dtype> &x, Vector<Dtype> &c, Vector<Dtype> &g,
  Vector<Dtype> &SubspaceMin) {

    // cached value: ThetaInverse=1/theta;
    Dtype theta_inverse = 1 / theta;

    // size of "t"
    std::vector<int> FreeVariablesIndex;
    // Debug(x_cauchy.transpose());

    //std::cout << "free vars " << FreeVariables.rows() << std::endl;
    for (int i = 0; i < x_cauchy.rows(); i++) {
      // Debug(x_cauchy(i) << " " << objFunc_->upperBound(i) << " " << objFunc_->lowerBound(i));
      if ((x_cauchy(i) != objFunc_->upperBound(i)) && (x_cauchy(i) != objFunc_->lowerBound(i))) {
        FreeVariablesIndex.push_back(i);
      }
    }
    const int FreeVarCount = FreeVariablesIndex.size();

    Matrix<Dtype> WZ = Matrix<Dtype>::Zero(W.cols(), FreeVarCount);

    for (int i = 0; i < FreeVarCount; i++)
      WZ.col(i) = W.row(FreeVariablesIndex[i]);

    // Debug(WZ);

    // r=(g+theta*(x_cauchy-x)-W*(M*c));
    // Debug(g);
    // Debug(x_cauchy);
    // Debug(x);
    Vector<Dtype> rr = (g + theta * (x_cauchy - x) - W * (M * c));
    // r=r(FreeVariables);
    Vector<Dtype> r = Matrix<Dtype>::Zero(FreeVarCount, 1);
    for (int i = 0; i < FreeVarCount; i++)
      r.row(i) = rr.row(FreeVariablesIndex[i]);

    // Debug(r.transpose());

    // STEP 2: "v = w^T*Z*r" and STEP 3: "v = M*v"
    Vector<Dtype> v = M * (WZ * r);
    // STEP 4: N = 1/theta*W^T*Z*(W^T*Z)^T
    Matrix<Dtype> N = theta_inverse * WZ * WZ.transpose();
    // N = I - MN
    N = Matrix<Dtype>::Identity(N.rows(), N.rows()) - M * N;
    // STEP: 5
    // v = N^{-1}*v
    v = N.lu().solve(v);
    // STEP: 6
    // HERE IS A MISTAKE IN THE ORIGINAL PAPER!
    Vector<Dtype> du = -theta_inverse * r
                       - theta_inverse * theta_inverse * WZ.transpose() * v;
    // Debug(du.transpose());
    // STEP: 7
    Dtype alpha_star = findAlpha(x_cauchy, du, FreeVariablesIndex);

    // STEP: 8
    Vector<Dtype> dStar = alpha_star * du;

    SubspaceMin = x_cauchy;
    for (int i = 0; i < FreeVarCount; i++) {
      SubspaceMin(FreeVariablesIndex[i]) = SubspaceMin(
                                             FreeVariablesIndex[i]) + dStar(i);
    }
  }

 public:
  void minimize(Problem<Dtype> &objFunc, Vector<Dtype> & x0) {
    objFunc_ = &objFunc;

    const size_t m = 10;

    DIM = x0.rows();

    std::cout << "-->> 1" << std::endl;

    if (!objFunc.hasLowerBound) {
      objFunc_->lowerBound = (-1 * Vector<Dtype>::Ones(DIM)) * std::numeric_limits<Dtype>::lowest();
      objFunc_->hasLowerBound = true;
    }

    if (!objFunc.hasUpperBound) {
      objFunc_->upperBound = Vector<Dtype>::Ones(DIM) * std::numeric_limits<Dtype>::max();
      objFunc_->hasUpperBound = true;
    }

    theta = 1.0;

    W = Matrix<Dtype>::Zero(DIM, 0);
    M = Matrix<Dtype>::Zero(0, 0);
    Matrix<Dtype> H = Matrix<Dtype>::Identity(DIM, DIM);

    std::cout << "-->> 3" << std::endl;

    xHistory.push_back(x0);
    std::cout << "-->> 4" << std::endl;

    Matrix<Dtype> yHistory = Matrix<Dtype>::Zero(DIM, 0);
    std::cout << "-->> 5" << std::endl;
    Matrix<Dtype> sHistory = Matrix<Dtype>::Zero(DIM, 0);
    std::cout << "-->> 6" << std::endl;

    Vector<Dtype> x = x0, g = x0;
    size_t k = 0;

    Dtype f = objFunc.value(x);
    std::cout << "-->> 7" << std::endl;
    objFunc.gradient(x, g);
    std::cout << "-->> 8" << std::endl;

    auto noConvergence =
    [&](Vector<Dtype> & x, Vector<Dtype> & g)->bool {
      return (((x - g).cwiseMax(objFunc_->lowerBound).cwiseMin(objFunc_->upperBound) - x).template lpNorm<Eigen::Infinity>() >= 1e-4);
    };

    while (noConvergence(x, g) && (k < this->settings_.maxIter)) {

      // Debug("iteration " << k)
      Dtype f_old = f;
      Vector<Dtype> x_old = x;
      Vector<Dtype> g_old = g;

      // STEP 2: compute the cauchy point by algorithm CP
      Vector<Dtype> CauchyPoint = Matrix<Dtype>::Zero(DIM, 1), c = Matrix<Dtype>::Zero(DIM, 1);
      GetGeneralizedCauchyPoint(x, g, CauchyPoint, c);
      // STEP 3: compute a search direction d_k by the primal method
      Vector<Dtype> SubspaceMin;
      SubspaceMinimization(CauchyPoint, x, c, g, SubspaceMin);

      Dtype Length = 0;
      // STEP 4: perform linesearch and STEP 5: compute gradient
      // WolfeRule::linesearch(x, SubspaceMin - x, FunctionValue, FunctionGradient);
      Dtype alpha_init = 1.0;
      const Dtype rate = MoreThuente<Dtype, decltype(objFunc), 1>::linesearch(x, -SubspaceMin + x,  objFunc, alpha_init);
      std::cout << "rate " << rate << std::endl;

      x = x - rate*(SubspaceMin - x);
      xHistory.push_back(x);

      // prepare for next iteration
      Vector<Dtype> newY = g - g_old;
      Vector<Dtype> newS = x - x_old;

      // STEP 6:
      Dtype test = newS.dot(newY);
      test = (test < 0) ? -1.0 * test : test;

      if (test > 1e-7 * newY.squaredNorm()) {
        if (k < this->settings_.m) {
          yHistory.conservativeResize(DIM, k + 1);
          sHistory.conservativeResize(DIM, k + 1);
        } else {

          yHistory.leftCols(this->settings_.m - 1) = yHistory.rightCols(
                                                this->settings_.m - 1).eval();
          sHistory.leftCols(this->settings_.m - 1) = sHistory.rightCols(
                                                this->settings_.m - 1).eval();
        }
        yHistory.rightCols(1) = newY;
        sHistory.rightCols(1) = newS;

        // STEP 7:
        theta = (Dtype)(newY.transpose() * newY)
                / (newY.transpose() * newS);

        W = Matrix<Dtype>::Zero(yHistory.rows(),
                         yHistory.cols() + sHistory.cols());

        W << yHistory, (theta * sHistory);

        Matrix<Dtype> A = sHistory.transpose() * yHistory;
        Matrix<Dtype> L = A.template triangularView<Eigen::StrictlyLower>();
        Matrix<Dtype> MM(A.rows() + L.rows(), A.rows() + L.cols());
        Matrix<Dtype> D = -1 * A.diagonal().asDiagonal();
        MM << D, L.transpose(), L, ((sHistory.transpose() * sHistory)
                                    * theta);

        M = MM.inverse();

      }

      Vector<Dtype> ttt = Matrix<Dtype>::Zero(1, 1);
      std::cout << objFunc(x) << std::endl;
      return;
      ttt(0) = f_old - f;
      // Debug("--> " << ttt.norm());
      if (ttt.norm() < 1e-8) {
        // successive function values too similar
        break;
      }
      k++;

    }

    x0 = x;

  }

};

}
/* namespace cppoptlib */

#endif /* LBFGSBSOLVER_H_ */
