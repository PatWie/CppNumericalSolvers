#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/bfgssolver.h"

// to use this library just use the namespace "cppoptlib"
namespace cppoptlib {

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
    cppoptlib::Vector<T> true_beta = cppoptlib::Vector<T>::Random(4);

    // create data
    cppoptlib::Matrix<T> X = cppoptlib::Matrix<T>::Random(50, 4);
    cppoptlib::Vector<T> y = 1.0/(1.0 + exp(-(X*true_beta).array()));

    // perform linear regression
    cppoptlib::LogisticRegression<T> f(X, y);

    cppoptlib::Vector<T> beta = cppoptlib::Vector<T>::Random(4);
    std::cout << "start in   " << beta.transpose() << std::endl;
    cppoptlib::BfgsSolver<double> solver;
    solver.minimize(f, beta);

    std::cout << "result     " << beta.transpose() << std::endl;
    std::cout << "true model " << true_beta.transpose() << std::endl;

    return 0;
}
