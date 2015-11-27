#include <iostream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/bfgssolver.h"

// to use CppNumericalSolvers just use the namespace "cppoptlib"
namespace cppoptlib {

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
    cppoptlib::Vector<T> true_beta = cppoptlib::Vector<T>::Random(4);

    // create data
    cppoptlib::Matrix<T> X = cppoptlib::Matrix<T>::Random(50, 4);
    cppoptlib::Vector<T> y = X*true_beta;

    // perform linear regression
    cppoptlib::LinearRegression<T> f(X, y);

    cppoptlib::Vector<T> beta = cppoptlib::Vector<T>::Random(4);
    std::cout << "start in   " << beta.transpose() << std::endl;
    cppoptlib::BfgsSolver<T> solver;
    solver.minimize(f, beta);

    std::cout << "result     " << beta.transpose() << std::endl;
    std::cout << "true model " << true_beta.transpose() << std::endl;

    return 0;
}
