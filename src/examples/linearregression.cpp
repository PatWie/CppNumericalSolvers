#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/bfgssolver.h"

// we define a new problem for optimizing the rosenbrock function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template<typename T>
class LinearRegression : public cppoptlib::Problem<T> {
  public:
    using typename cppoptlib::Problem<T>::TVector;
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  protected:
    const MatrixType X;
    const TVector y;
    const MatrixType XX;

  public:
    LinearRegression(const MatrixType &X_, const TVector &y_) : X(X_), y(y_), XX(X_.transpose()*X_) {}

    T value(const TVector &beta) {
        return 0.5*(X*beta-y).squaredNorm();
    }

    void gradient(const TVector &beta, TVector &grad) {
        grad = XX*beta - X.transpose()*y;
    }
};

int main(int argc, char const *argv[]) {
    typedef LinearRegression<double> TLinearRegression;
    typedef typename TLinearRegression::TVector TVector;
    typedef typename TLinearRegression::MatrixType MatrixType;

    // create true model
    TVector true_beta = TVector::Random(4);

    // create data
    MatrixType X = MatrixType::Random(50, 4);
    TVector y = X*true_beta;

    // perform linear regression
    TLinearRegression f(X, y);

    TVector beta = TVector::Random(4);
    std::cout << "start in   " << beta.transpose() << std::endl;
    cppoptlib::BfgsSolver<TLinearRegression> solver;
    solver.minimize(f, beta);

    std::cout << "result     " << beta.transpose() << std::endl;
    std::cout << "true model " << true_beta.transpose() << std::endl;

    return 0;
}
