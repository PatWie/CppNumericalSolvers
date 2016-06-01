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
  Vector<Dtype> lboundTemplate;
  Vector<Dtype> uboundTemplate;
  Dtype theta;
  int DIM;
  int m_historySize = 5;

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
    Vector<Dtype> d = -g;
    // n operations
    for (int j = 0; j < DIM; j++) {
      if (g(j) == 0) {
        SetOfT.push_back(std::make_pair(j, std::numeric_limits<Dtype>::max()));
      } else {
        Dtype tmp = 0;
        if (g(j) < 0) {
          tmp = (x(j) - uboundTemplate(j)) / g(j);
        } else {
          tmp = (x(j) - lboundTemplate(j)) / g(j);
        }
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
      if (SetOfT[sortedIndices[j]].second > 0)
        break;
    }
    int b = sortedIndices[i];
    // see below
    // t                    :=  min{t_i : i in F}
    Dtype t = SetOfT[b].second;
    // \delta Dtype             :=  t - 0
    Dtype dt = t ;
    // examination of subsequent segments
    while ((dt_min >= dt) && (i < DIM)) {
      if (d(b) > 0)
        x_cauchy(b) = uboundTemplate(b);
      else if (d(b) < 0)
        x_cauchy(b) = lboundTemplate(b);
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
    #pragma omp parallel for if (x_cauchy.rows() > 1000)
    for (int ii = i; ii < x_cauchy.rows(); ii++) {
      x_cauchy(sortedIndices[ii]) = x(sortedIndices[ii]) + t_old * d(sortedIndices[ii]);
    }
    c += dt_min * p;
  }
  /**
   * @brief find alpha* = max {a : a <= 1 and  l_i-xc_i <= a*d_i <= u_i-xc_i}
   * @details [long description]
   *
   * @param FreeVariables [description]
   * @return [description]
   */
  Dtype findAlpha(Vector<Dtype> &x_cp, Vector<Dtype> &du, std::vector<int> &FreeVariables) {
    Dtype alphastar = 1;
    const unsigned int n = FreeVariables.size();
    for (unsigned int i = 0; i < n; i++) {
      if (du(i) > 0) {
        alphastar = std::min(alphastar, (uboundTemplate(FreeVariables[i]) - x_cp(FreeVariables[i])) / du(i));
      } else {
        alphastar = std::min(alphastar, (lboundTemplate(FreeVariables[i]) - x_cp(FreeVariables[i])) / du(i));
      }
    }
    return alphastar;
  }
  /**
   * @brief solving unbounded probelm
   * @details [long description]
   *
   * @param SubspaceMin [description]
   */
  void SubspaceMinimization(Vector<Dtype> &x_cauchy, Vector<Dtype> &x, Vector<Dtype> &c, Vector<Dtype> &g,
  Vector<Dtype> &SubspaceMin) {
    Dtype theta_inverse = 1 / theta;
    std::vector<int> FreeVariablesIndex;
    for (int i = 0; i < x_cauchy.rows(); i++) {
      if ((x_cauchy(i) != uboundTemplate(i)) && (x_cauchy(i) != lboundTemplate(i))) {
        FreeVariablesIndex.push_back(i);
      }
    }
    const int FreeVarCount = FreeVariablesIndex.size();
    Matrix<Dtype> WZ = Matrix<Dtype>::Zero(W.cols(), FreeVarCount);
    for (int i = 0; i < FreeVarCount; i++)
      WZ.col(i) = W.row(FreeVariablesIndex[i]);
    Vector<Dtype> rr = (g + theta * (x_cauchy - x) - W * (M * c));
    // r=r(FreeVariables);
    Vector<Dtype> r = Matrix<Dtype>::Zero(FreeVarCount, 1);
    for (int i = 0; i < FreeVarCount; i++)
      r.row(i) = rr.row(FreeVariablesIndex[i]);
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
    Vector<Dtype> du = -theta_inverse * r - theta_inverse * theta_inverse * WZ.transpose() * v;
    // STEP: 7
    Dtype alpha_star = findAlpha(x_cauchy, du, FreeVariablesIndex);
    // STEP: 8
    Vector<Dtype> dStar = alpha_star * du;
    SubspaceMin = x_cauchy;
    for (int i = 0; i < FreeVarCount; i++) {
      SubspaceMin(FreeVariablesIndex[i]) = SubspaceMin(FreeVariablesIndex[i]) + dStar(i);
    }
  }
 public:
  void setHistorySize(const int hs) { m_historySize = hs; }

  void minimize(Problem<Dtype> &objFunc, Vector<Dtype> & x0) {
    objFunc_ = &objFunc;
    DIM = x0.rows();
    if (objFunc.hasLowerBound()) {
      lboundTemplate = objFunc_->lowerBound();
    }else {
      lboundTemplate = -Vector<Dtype>::Ones(DIM)* std::numeric_limits<Dtype>::infinity();
    }
    if (objFunc.hasUpperBound()) {
      uboundTemplate = objFunc_->upperBound();
    }else {
      uboundTemplate = Vector<Dtype>::Ones(DIM)* std::numeric_limits<Dtype>::infinity();
    }
    theta = 1.0;
    W = Matrix<Dtype>::Zero(DIM, 0);
    M = Matrix<Dtype>::Zero(0, 0);
    xHistory.push_back(x0);
    Matrix<Dtype> yHistory = Matrix<Dtype>::Zero(DIM, 0);
    Matrix<Dtype> sHistory = Matrix<Dtype>::Zero(DIM, 0);
    Vector<Dtype> x = x0, g = x0;
    Dtype f = objFunc.value(x);
    objFunc.gradient(x, g);
    // conv. crit.
    auto noConvergence =
    [&](Vector<Dtype> & x, Vector<Dtype> & g)->bool {
      return (((x - g).cwiseMax(lboundTemplate).cwiseMin(uboundTemplate) - x).template lpNorm<Eigen::Infinity>() >= 1e-4);
    };
    this->m_current.reset();
    this->m_status = Status::Continue;
    while (objFunc.callback(this->m_current, x) && noConvergence(x, g) && (this->m_status == Status::Continue)) {
      Dtype f_old = f;
      Vector<Dtype> x_old = x;
      Vector<Dtype> g_old = g;
      // STEP 2: compute the cauchy point
      Vector<Dtype> CauchyPoint = Matrix<Dtype>::Zero(DIM, 1), c = Matrix<Dtype>::Zero(DIM, 1);
      GetGeneralizedCauchyPoint(x, g, CauchyPoint, c);
      // STEP 3: compute a search direction d_k by the primal method for the sub-problem
      Vector<Dtype> SubspaceMin;
      SubspaceMinimization(CauchyPoint, x, c, g, SubspaceMin);
      // STEP 4: perform linesearch and STEP 5: compute gradient
      Dtype alpha_init = 1.0;
      const Dtype rate = MoreThuente<Dtype, decltype(objFunc), 1>::linesearch(x,  SubspaceMin-x ,  objFunc, alpha_init);
      // update current guess and function information
      x = x - rate*(x-SubspaceMin);
      f = objFunc.value(x);
      objFunc.gradient(x, g);
      xHistory.push_back(x);
      // prepare for next iteration
      Vector<Dtype> newY = g - g_old;
      Vector<Dtype> newS = x - x_old;
      // STEP 6:
      Dtype test = newS.dot(newY);
      test = (test < 0) ? -1.0 * test : test;
      if (test > 1e-7 * newY.squaredNorm()) {
        if (yHistory.cols() < m_historySize) {
          yHistory.conservativeResize(DIM, this->m_current.iterations + 1);
          sHistory.conservativeResize(DIM, this->m_current.iterations + 1);
        } else {
          yHistory.leftCols(m_historySize - 1) = yHistory.rightCols(m_historySize - 1).eval();
          sHistory.leftCols(m_historySize - 1) = sHistory.rightCols(m_historySize - 1).eval();
        }
        yHistory.rightCols(1) = newY;
        sHistory.rightCols(1) = newS;
        // STEP 7:
        theta = (Dtype)(newY.transpose() * newY) / (newY.transpose() * newS);
        W = Matrix<Dtype>::Zero(yHistory.rows(), yHistory.cols() + sHistory.cols());
        W << yHistory, (theta * sHistory);
        Matrix<Dtype> A = sHistory.transpose() * yHistory;
        Matrix<Dtype> L = A.template triangularView<Eigen::StrictlyLower>();
        Matrix<Dtype> MM(A.rows() + L.rows(), A.rows() + L.cols());
        Matrix<Dtype> D = -1 * A.diagonal().asDiagonal();
        MM << D, L.transpose(), L, ((sHistory.transpose() * sHistory) * theta);
        M = MM.inverse();
      }
      Vector<Dtype> ttt = Matrix<Dtype>::Zero(1, 1);
      ttt(0) = f_old - f;
      if (ttt.norm() < 1e-8) {
        // successive function values too similar
        break;
      }
      ++this->m_current.iterations;
      this->m_current.gradNorm = g.norm();
      this->m_status = checkConvergence(this->m_stop, this->m_current);
    }
    x0 = x;
    if (this->m_debug > DebugLevel::None) {
        std::cout << "Stop status was: " << this->m_status << std::endl;
        std::cout << "Stop criteria were: " << std::endl << this->m_stop << std::endl;
        std::cout << "Current values are: " << std::endl << this->m_current << std::endl;
    }
  }
};
}
/* namespace cppoptlib */
#endif /* LBFGSBSOLVER_H_ */
