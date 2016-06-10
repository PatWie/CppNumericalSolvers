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
template<typename ProblemType>
class CMAesSolver : public ISolver<ProblemType, 1> {
  public:
    using Superclass = ISolver<ProblemType, 1>;
    using typename Superclass::Scalar;
    using typename Superclass::TVector;
    using typename Superclass::THessian;

  protected:
  // random number generator
  // http://stackoverflow.com/questions/14732132/global-initialization-with-temporary-function-object
  // we construct this in the constructor
  // std::mt19937 gen {std::random_device {}()};
  std::mt19937 gen;
  // each sample from population
  struct individual {
    TVector pos;
    TVector step;
    Scalar cost;

    individual(int n) {
      pos = TVector::Zero(n);
      step = TVector::Zero(n);
    }
    individual() {}

    void reset(int n) {
      pos = TVector::Zero(n);
      step = TVector::Zero(n);
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
  TVector sampleMvn(TVector &mean, THessian &covar) {

    THessian normTransform;
    Eigen::LLT<THessian> cholSolver(covar);

    if (cholSolver.info() == Eigen::Success) {
      normTransform = cholSolver.matrixL();
    } else {
      Eigen::SelfAdjointEigenSolver<THessian> eigenSolver(covar);
      normTransform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    TVector stdNormDistr = TVector::Zero(mean.rows());
    std::normal_distribution<> d(0, 1);
    for (int i = 0; i < mean.rows(); ++i) {
      stdNormDistr[i] = d(gen);
    }

    TVector samples = normTransform * stdNormDistr + mean;

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
  void minimize(ProblemType &objFunc, TVector &x0) {

    const int DIM = x0.rows();

    // start from initial guess
    individual ii(DIM);
    ii.pos = x0;

    Scalar VarMin = -DIM;
    Scalar VarMax = DIM;

    const int populationSize = (4 + round(3 * log(DIM))) * 10;
    const int mu = round(populationSize / 2);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> w = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(mu);
    for (int i = 1; i <= mu; ++i) {
      w(i - 1) = log(mu + 0.5) - log(i);
    }

    w /= w.sum();

    const Scalar mu_eff   = 1. / w.dot(w);

    const Scalar sigma0   = 0.3 * (VarMax - VarMin);
    const Scalar cs       = (mu_eff + 2.) / (DIM + mu_eff + 5.);
    const Scalar ds       = 1. + cs + 2.*std::max(sqrt((mu_eff - 1) / (DIM + 1)) - 1, Scalar(0.));
    const Scalar ENN      = sqrt(DIM) * (1 - 1. / (4.*DIM) + 1. / (21.*DIM * DIM));

    const Scalar cc       = (4. + mu_eff / DIM) / (4. + DIM + 2.*mu_eff / DIM);
    const Scalar c1       = 2. / ((DIM + 1.3) * (DIM + 1.3) + mu_eff);
    const Scalar alpha_mu = 2.;
    const Scalar cmu      = std::min(1. - c1, alpha_mu * (mu_eff - 2. + 1. / mu_eff) / ((DIM + 2.) * (DIM + 2.) + alpha_mu * mu_eff / 2.));
    const Scalar hth      = (1.4 + 2 / (DIM + 1.)) * ENN;

    TVector ps;
    TVector pc;
    THessian C;
    Scalar sigma = sigma0;

    individual M;

    TVector t = TVector::Zero(DIM);
    ps = t;
    pc = t;
    THessian eye = THessian::Identity(DIM, DIM);
    C = eye;

    std::uniform_real_distribution<> dis(VarMin, VarMax);
    M.pos = TVector::Zero(DIM);
    for (int i = 0; i < DIM; ++i) {
      M.pos[i] = dis(gen);
    }

    M.step = TVector::Zero(DIM);
    M.cost = objFunc(M.pos);

    individual bestSol = M;

    Scalar bestCostSoFar;

    TVector zeroVectorTemplate = TVector::Zero(DIM);

    // CMA-ES Main Loop
    for (size_t curIter = 0; curIter < this->m_stop.iterations; ++curIter) {
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
      if (curIter == this->m_stop.iterations - 2)
        break;

      // update mean (TODO: matrix-vec-multiplication with permutation matrix?)
      M.step = TVector::Zero(DIM);
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

      Scalar hs = 0;
      if (ps.norm() / sqrt(pow(1 - (1. - cs), (2 * (curIter + 1)))) < hth)
        hs = 1;
      else
        hs = 0;

      const Scalar delta = (1 - hs) * cc * (2 - cc);

      pc = (1 - cc) * pc + hs * sqrt(cc * (2. - cc) * mu_eff) * M.step;
      C = (1 - c1 - cmu) * C + c1 * (pc * pc.transpose() + delta * C);

      for (int j = 0; j < mu; ++j) {
        C += cmu * w(j) * pop[j].step * pop[j].step.transpose();
      }

      Eigen::EigenSolver<THessian> eig(C);
      TVector E = eig.eigenvalues().real();
      THessian V = eig.eigenvectors().real();

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

      if ((curIter > 150) && ((bestSol.pos-x0).norm() < 1e-8)) {
        // successive function values too similar, but enought pre-iteration
        break;
      }

      // update best solution
      x0 = bestSol.pos;

      if(!objFunc.callback(this->m_current, x0))
        break;
    }
  }

};

} /* namespace cppoptlib */

#endif /* CMAES_H_ */
