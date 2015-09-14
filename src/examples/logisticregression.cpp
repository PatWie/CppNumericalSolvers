#include <iostream>
#include "../../include/cns/meta.h"
#include "../../include/cns/problem.h"
#include "../../include/cns/solver/bfgssolver.h"

// to use CppNumericalSolvers just use the namespace "cns"
namespace cns {

// we define a new problem for optimizing the rosenbrock function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template<typename T>
class LogisticRegression : public Problem<T> {
    const Matrix<T> X;
    const Vector<T> y;
    const Matrix<T> XX;

  public:
    LogisticRegression(const Matrix<T> &X_, const Vector<T> y_) : X(X_), y(y_), XX(X_.transpose()*X_) {}

    T value(const Vector<T> &beta) {
        return (1.0/(1.0 + exp(-(X*beta).array())) - y.array()).matrix().squaredNorm();
    }

    void gradient(const Vector<T> &beta, Vector<T> &grad) {
        const Vector<T> p = 1.0/(1.0 + exp(-(X*beta).array()));
        grad = X.transpose()*(p-y);
    }
};

}
int main(int argc, char const *argv[]) {
    typedef double T;
    srand((unsigned int) time(0));

    // create true model
    cns::Vector<T> true_beta = cns::Vector<T>::Random(4);

    // create data
    cns::Matrix<T> X = cns::Matrix<T>::Random(50, 4);
    cns::Vector<T> y = 1.0/(1.0 + exp(-(X*true_beta).array()));

    // perform linear regression
    cns::LogisticRegression<T> f(X, y);

    cns::Vector<T> beta = cns::Vector<T>::Random(4);
    std::cout << "start in   " << beta.transpose() << std::endl;
    cns::BfgsSolver<double> solver;
    solver.minimize(f, beta);

    std::cout << "result     " << beta.transpose() << std::endl;
    std::cout << "true model " << true_beta.transpose() << std::endl;

    return 0;
}