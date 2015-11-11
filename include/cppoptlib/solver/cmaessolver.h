// CppNumericalSolver
#ifndef CMAES_H_
#define CMAES_H_

#include <random>
#include <Eigen/Dense>
#include "isolver.h"

namespace cppoptlib {

template<typename T>
class CMAesSolver : public ISolver<T, 1> {

  std::mt19937 *gen;

  struct individual {
    Vector<T> pos;
    Vector<T> step;
    T cost;

    individual(int n) {
      pos = Vector<T>::Zero(n);
      step = Vector<T>::Zero(n);
    }
    individual() {}

    void reset(int n) {
      pos = Vector<T>::Zero(n);
      step = Vector<T>::Zero(n);
    }

    T costValue() {
      return 20 * (1 - exp(-0.2 * sqrt(pos.dot(pos) / pos.rows()))) + exp(1.) - exp(cos((2 * 3.14159265358979 * pos).mean()));
    }

    T update() {
      cost = 20 * (1 - exp(-0.2 * sqrt(pos.dot(pos) / pos.rows()))) + exp(1.) - exp(cos((2 * 3.14159265358979 * pos).mean()));
    }
  };

  Vector<T> sampleMvn(int n, T sig) {
    Vector<T> ans = Vector<T>::Zero(n);
    std::normal_distribution<> d(0, sig);
    for (int i = 0; i < n; ++i) {
      ans[i] = d(*gen);
    }
    return ans;
  }

  Vector<T> sampleMvn(Vector<T> &mean, Matrix<T> &covar) {

    Matrix<T> normTransform;
    Eigen::LLT<Eigen::MatrixXd> cholSolver(covar);

    if (cholSolver.info() == Eigen::Success) {
      // Use cholesky solver
      normTransform = cholSolver.matrixL();
    } else {
      // Use eigen solver
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
      normTransform = eigenSolver.eigenvectors()
                      * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Vector<T> ans = Vector<T>::Zero(mean.rows());
    std::normal_distribution<> d(0, 1);
    for (int i = 0; i < mean.rows(); ++i) {
      ans[i] = d(*gen);
    }

    Eigen::MatrixXd samples = normTransform
                              * ans
                              + mean;

    return samples;
  }

 public:

  CMAesSolver() {

    std::random_device rd;
    gen = new std::mt19937(rd());
  }

  ~CMAesSolver() {
    std::cout << "destruct" << std::endl;

    // gen = nullptr;
  }
  /**
   * @brief minimize
   * @details [long description]
   *
   * @param objFunc [description]
   */
  void minimize(Problem<T> &objFunc, Vector<T> & x0) {
    const int DIM = x0.rows();
    int nVar = DIM;

    individual ii(nVar);
    ii.pos[0] = 0;
    ii.pos[1] = 1;
    ii.pos[2] = 2;
    ii.pos[3] = 3;
    ii.pos[4] = 4;

    T VarMin = -10;
    T VarMax = 10;

    int maxIter = 300;

    int lambda = (4 + round(3 * log(DIM))) * 10;
    int mu = round(lambda / 2);

    Vector<T> w = Vector<T>::Zero(mu);
    for (int i = 1; i <= mu; ++i) {
      w(i - 1) = log(mu + 0.5) - log(i);
    }

    w /= w.sum();

    T mu_eff = 1. / w.dot(w);

    T sigma0 = 0.3 * (VarMax - VarMin);
    T cs = (mu_eff + 2.) / (nVar + mu_eff + 5.);
    T ds = 1. + cs + 2.*std::max(sqrt((mu_eff - 1) / (nVar + 1)) - 1, (T)0.);
    T ENN = sqrt(nVar) * (1 - 1. / (4.*nVar) + 1. / (21.*nVar * nVar));

    T cc = (4. + mu_eff / nVar) / (4. + nVar + 2.*mu_eff / nVar);
    T c1 = 2. / ((nVar + 1.3) * (nVar + 1.3) + mu_eff);
    T alpha_mu = 2.;
    T cmu = std::min(1. - c1, alpha_mu * (mu_eff - 2. + 1. / mu_eff) / ((nVar + 2.) * (nVar + 2.) + alpha_mu * mu_eff / 2.));
    T hth = (1.4 + 2 / (nVar + 1.)) * ENN;

    std::vector<Vector<T> >ps; ps.resize(maxIter);
    std::vector<Vector<T> >pc; pc.resize(maxIter);
    std::vector<Matrix<T> >C;  C.resize(maxIter);
    std::vector<T >sigma; sigma.resize(lambda);
    sigma[0] = sigma0;
    std::vector<individual> M;

    Vector<T> t = Vector<T>::Zero(nVar);
    ps[0] = t;
    pc[0] = t;
    Matrix<T> eye = Matrix<T>::Identity(nVar, nVar);
    C[0] = eye;

    M.resize(maxIter);
    std::uniform_real_distribution<> dis(VarMin, VarMax);
    M[0].pos = Vector<T>::Zero(nVar);
    for (int i = 0; i < nVar; ++i) {
      M[0].pos[i] = dis(*gen);
    }
    std::cout << M[0].pos.transpose() << std::endl;

    M[0].step = Vector<T>::Zero(nVar);
    M[0].update();

    individual bestSol = M[0];

    std::vector<T> bestCost;
    bestCost.resize(maxIter);

    Vector<T> zero = Vector<T>::Zero(nVar);
    // CMA-ES Main Loop
    for (int g = 0; g < maxIter - 1; ++g) {
      std::vector<individual> pop;

      for (int i = 0; i < lambda; ++i) {
        individual curInd;
        curInd.step = sampleMvn(zero, C[g]).eval();

        curInd.pos = M[g].pos + sigma[g] * curInd.step;
        curInd.cost = curInd.costValue();

        if (curInd.cost < bestSol.cost) {
          bestSol = curInd;
        }

        pop.push_back(curInd);

      }

      sort(pop.begin(), pop.end(), [&](const individual & a, const  individual & b) -> bool {
        return a.cost < b.cost;
      });

      bestCost[g] = bestSol.cost;
      printf("%i best cost %f\n", g, bestCost[g] );

      if (g == maxIter - 2)
        break;

      // update mean
      M[g + 1].step = Vector<T>::Zero(nVar);
      for (int j = 0; j < mu; ++j) {
        M[g + 1].step += w[j] * pop[j].step;
      }

      M[g + 1].pos = M[g].pos + sigma[g] * M[g + 1].step;
      M[g + 1].update();
      if (M[g + 1].cost < bestSol.cost)
        bestSol = M[g + 1];

      // Update Step Size

      ps[g + 1] = (1. - cs) * ps[g] + sqrt(cs * (2. - cs) * mu_eff) * C[g].llt().matrixL().transpose().solve(M[g + 1].step);
      sigma[g + 1] = sigma[g] * pow(  exp((cs / ds * ((ps[g + 1]).norm() / ENN - 1.))), 0.3);

      T hs = 0;
      if (ps[g + 1].norm() / sqrt(pow(1 - (1. - cs), (2 * (g + 1)))) < hth)
        hs = 1;
      else
        hs = 0;

      T delta = (1 - hs) * cc * (2 - cc);

      pc[g + 1] = (1 - cc) * pc[g] + hs * sqrt(cc * (2. - cc) * mu_eff) * M[g + 1].step;
      C[g + 1] = (1 - c1 - cmu) * C[g] + c1 * (pc[g + 1] * pc[g + 1].transpose() + delta * C[g]);

      for (int j = 0; j < mu; ++j) {
        C[g + 1] += cmu * w(j) * pop[j].step * pop[j].step.transpose();
      }

      Eigen::EigenSolver<Matrix<T> > eig(C[g + 1]);
      Vector<T> E = eig.eigenvalues().real();
      Matrix<T> V = eig.eigenvectors().real();

      bool pd = true;
      for (int i = 0; i < nVar; ++i) {
        if (E(i) < 0) {
          pd = false;
          E = E.cwiseMax(zero);
          C[g + 1] = V * E.asDiagonal() * V.inverse();
          break;
        }
      }

    }
    std::cout << "outloop" << std::endl;

  }

};

} /* namespace cppoptlib */

#endif /* CMAES_H_ */
