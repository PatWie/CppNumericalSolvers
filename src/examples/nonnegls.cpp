#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/lbfgsbsolver.h"

// to use CppNumericalSolvers just use the namespace "cppoptlib"
namespace cppoptlib {

// we will solve ||Xb-y|| s.t. b>=0
template<typename T, int D>
class NonNegativeLeastSquares : public Problem<T, D> {
  public:
    using typename Problem<T, D>::TVector;
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    const MatrixType X;
    const TVector y;

  public:
    NonNegativeLeastSquares(const MatrixType &X_, const TVector y_) : X(X_), y(y_) {}

    T value(const TVector &beta) {
        return (X*beta-y).dot(X*beta-y);
    }

    void gradient(const TVector &beta, TVector &grad) {
        grad = X.transpose()*2*(X*beta-y);
    }
};

}
int main(int argc, char const *argv[]) {

    const size_t DIM = 4;
    const size_t NUM = 5;
    typedef double T;
    typedef cppoptlib::NonNegativeLeastSquares<T, DIM> TNNLS;
    typedef typename TNNLS::TVector TVector;
    typedef typename TNNLS::MatrixType MatrixType;

    // create model X*b for arbitrary b
    MatrixType X         = MatrixType::Random(NUM, DIM);
    TVector true_beta = TVector::Random();
    MatrixType y         = X*true_beta;

    // perform non-negative least squares
    TNNLS f(X, y);
    f.setLowerBound(TVector::Zero());

    // create initial guess (make sure it's valid >= 0)
    TVector beta = TVector::Random();
    beta = (beta.array() < 0).select(-beta, beta);
    std::cout << "start with b =          " << beta.transpose() << std::endl;

    // init L-BFGS-B for box-constrained solving
    cppoptlib::LbfgsbSolver<TNNLS> solver;
    solver.minimize(f, beta);

    // display results
    std::cout << "model s.t. b >= 0  loss:" << f(beta) << std::endl;
    std::cout << "for b =                 " << beta.transpose() << std::endl;
    std::cout << "true model         loss:" << f(true_beta) << std::endl;
    std::cout << "for b =                 " << true_beta.transpose() << std::endl;

    return 0;
}
