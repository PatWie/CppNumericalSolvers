#include <iostream>
#include "../../include/cns/meta.h"
#include "../../include/cns/problem.h"
#include "../../include/cns/solver/bfgssolver.h"

// to use CppNumericalSolvers just use the namespace "cns"
namespace cns {

// we define a new problem for optimizing the rosenbrock function
// we use a templated-class rather than "auto"-lambda function for a clean architecture
template<typename T>
class LinearRegression : public Problem<T> {
    const Matrix<T> X;
    const Vector<T> y;
    const Matrix<T> XX;

  public:
    LinearRegression(const Matrix<T> &X_, const Vector<T> y_) : X(X_), y(y_), XX(X_.transpose()*X_) {}

    T value(const Vector<T> &beta) {
        return 0.5*(X*beta-y).squaredNorm();
    }

    void gradient(const Vector<T> &beta, Vector<T> &grad) {
        grad = XX*beta - X.transpose()*y;
    }
};

}
int main(int argc, char const *argv[]) {
    typedef double T;

    // create true model
    cns::Vector<T> true_beta = cns::Vector<T>::Random(4);

    // create data
    cns::Matrix<T> X = cns::Matrix<T>::Random(50, 4);
    cns::Vector<T> y = X*true_beta;

    // perform linear regression
    cns::LinearRegression<T> f(X, y);

    cns::Vector<T> beta = cns::Vector<T>::Random(4);
    std::cout << "start in   " << beta.transpose() << std::endl;
    cns::BfgsSolver<double> solver;
    solver.minimize(f, beta);

    std::cout << "result     " << beta.transpose() << std::endl;
    std::cout << "true model " << true_beta.transpose() << std::endl;

    return 0;
}