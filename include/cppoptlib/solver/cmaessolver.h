// CppNumericalSolver
#ifndef CMAES_H_
#define CMAES_H_

#include <random>
#include <Eigen/Dense>
#include "isolver.h"

namespace cppoptlib {

/**
 * @brief Covariance Matrix Adaptation
 */
template<typename T>
class CMAesSolver : public ISolver<T, 1> {
  // random number generator
  // http://stackoverflow.com/questions/14732132/global-initialization-with-temporary-function-object
  // we construct this in the constructor
  // std::mt19937 gen {std::random_device {}()};
  std::mt19937 gen;
  // each sample from population
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
  };

  /**
   * @brief sample from MVN with given mean and covmat
   * @details [long description]
   *
   * @param mean mean of distribution
   * @param covar covariance
   * @return [description]
   */
  Vector<T> sampleMvn(Vector<T> &mean, Matrix<T> &covar) {

    Matrix<T> normTransform;
    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > cholSolver(covar);

    if (cholSolver.info() == Eigen::Success) {
      normTransform = cholSolver.matrixL();
    } else {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > eigenSolver(covar);
      normTransform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Vector<T> stdNormDistr = Vector<T>::Zero(mean.rows());
    std::normal_distribution<> d(0, 1);
    for (int i = 0; i < mean.rows(); ++i) {
      stdNormDistr[i] = d(gen);
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> samples = normTransform * stdNormDistr + mean;

    return samples;
  }

 public:

  CMAesSolver() : gen((std::random_device())()) {

  }

  /**
   * @brief minimize
   * @details [long description]
   *
   * @param objFunc [description]
   */
  void minimize(Problem<T> &objFunc, Vector<T> & x0) {

    const int DIM = x0.rows();

    // start from initial guess
    individual ii(DIM);
    ii.pos = x0;

    T VarMin = -DIM;
    T VarMax = DIM;

    const int populationSize = (4 + round(3 * log(DIM))) * 10;
    const int mu = round(populationSize / 2);

    Vector<T> w = Vector<T>::Zero(mu);
    for (int i = 1; i <= mu; ++i) {
      w(i - 1) = log(mu + 0.5) - log(i);
    }

    w /= w.sum();

    const T mu_eff   = 1. / w.dot(w);

    const T sigma0   = 0.3 * (VarMax - VarMin);
    const T cs       = (mu_eff + 2.) / (DIM + mu_eff + 5.);
    const T ds       = 1. + cs + 2.*std::max((T)sqrt((mu_eff - 1) / (DIM + 1)) - 1, (T)0.);
    const T ENN      = sqrt(DIM) * (1 - 1. / (4.*DIM) + 1. / (21.*DIM * DIM));

    const T cc       = (4. + mu_eff / DIM) / (4. + DIM + 2.*mu_eff / DIM);
    const T c1       = 2. / ((DIM + 1.3) * (DIM + 1.3) + mu_eff);
    const T alpha_mu = 2.;
    const T cmu      = std::min(1. - c1, alpha_mu * (mu_eff - 2. + 1. / mu_eff) / ((DIM + 2.) * (DIM + 2.) + alpha_mu * mu_eff / 2.));
    const T hth      = (1.4 + 2 / (DIM + 1.)) * ENN;

    Vector<T> ps;
    Vector<T> pc;
    Matrix<T> C;
    T sigma = sigma0;

    individual M;

    Vector<T> t = Vector<T>::Zero(DIM);
    ps = t;
    pc = t;
    Matrix<T> eye = Matrix<T>::Identity(DIM, DIM);
    C = eye;

    std::uniform_real_distribution<> dis(VarMin, VarMax);
    M.pos = Vector<T>::Zero(DIM);
    for (int i = 0; i < DIM; ++i) {
      M.pos[i] = dis(gen);
    }

    M.step = Vector<T>::Zero(DIM);
    M.cost = objFunc(M.pos);

    individual bestSol = M;

    T bestCostSoFar;

    Vector<T> zeroVectorTemplate = Vector<T>::Zero(DIM);

    // CMA-ES Main Loop
    for (this->iterations_ = 0; this->iterations_ < this->settings_.maxIter; ++this->iterations_) {
      std::vector<individual> pop;

      for (int i = 0; i < populationSize; ++i) {
        individual curInd;
        curInd.step = sampleMvn(zeroVectorTemplate, C).eval();
        curInd.pos = M.pos + sigma * curInd.step;
        curInd.cost = objFunc(curInd.pos);

        if (curInd.cost < bestSol.cost) {
          bestSol = curInd;
        }
        pop.push_back(curInd);

      }

      // sort them according their fitness
      sort(pop.begin(), pop.end(), [&](const individual & a, const  individual & b) -> bool {
        return a.cost < b.cost;
      });

      bestCostSoFar = bestSol.cost;
      // printf("%i best cost so far %f\n", curIter, bestCostSoFar );

      // any further update?
      if (this->iterations_ == this->settings_.maxIter - 2)
        break;

      // update mean (TODO: matrix-vec-multiplication with permutation matrix?)
      M.step = Vector<T>::Zero(DIM);
      for (int j = 0; j < mu; ++j) {
        M.step += w[j] * pop[j].step;
      }

      // shift current position
      M.pos = M.pos + sigma * M.step;
      M.cost = objFunc(M.pos);
      if (M.cost < bestSol.cost)
        bestSol = M;

      // update step size
      ps = (1. - cs) * ps + sqrt(cs * (2. - cs) * mu_eff) * C.llt().matrixL().transpose().solve(M.step);
      sigma = sigma * pow(  exp((cs / ds * ((ps).norm() / ENN - 1.))), 0.3);

      T hs = 0;
      if (ps.norm() / sqrt(pow(1 - (1. - cs), (2 * (curIter + 1)))) < hth)
        hs = 1;
      else
        hs = 0;

      const T delta = (1 - hs) * cc * (2 - cc);

      pc = (1 - cc) * pc + hs * sqrt(cc * (2. - cc) * mu_eff) * M.step;
      C = (1 - c1 - cmu) * C + c1 * (pc * pc.transpose() + delta * C);

      for (int j = 0; j < mu; ++j) {
        C += cmu * w(j) * pop[j].step * pop[j].step.transpose();
      }

      Eigen::EigenSolver<Matrix<T> > eig(C);
      Vector<T> E = eig.eigenvalues().real();
      Matrix<T> V = eig.eigenvectors().real();

      // check positive definitness of covariance matrix (all eigenvalues must be > 0)
      bool pd = true;
      for (int i = 0; i < DIM; ++i) {
        if (E(i) < 0) {
          // Oops, Eigen value to small
          pd = false;
          E = E.cwiseMax(zeroVectorTemplate);
          C = V * E.asDiagonal() * V.inverse();
          break;
        }
      }

      // prepare new population
      pop.clear();

      if ((this->iterations_ > 150) && ((bestSol.pos-x0).norm() < 1e-8)) {
        // successive function values too similar, but enought pre-iteration
        break;
      }

      // update best solution
      x0 = bestSol.pos;
    }
  }

};

} /* namespace cppoptlib */

#endif /* CMAES_H_ */
