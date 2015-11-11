#include <iostream>
#include <fstream>
#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"
#include "../../include/cppoptlib/solver/cmaessolver.h"

// to use this library just use the namespace "cppoptlib"
namespace cppoptlib {

template<typename T>
class NonNegLS : public Problem<T> {
  public:
    Matrix<T> X;
    Vector<T> y;
    // this is just the objective (NOT optional)
    T value(const Vector<T> &w) {
        return (X*w-y).dot(X*w-y);
    }

    // if you calculated the derivative by hand
    // you can implement it here (OPTIONAL)
    // otherwise it will fall back to (bad) numerical finite differences
    void gradient(const Vector<T> &w, Vector<T> &grad) {
        grad = X.transpose()*2*(X*w-y);
    }

};

}

int main(int argc, char const *argv[]) {
    cppoptlib::NonNegLS<double> f;
    cppoptlib::Vector<double> beta = cppoptlib::Vector<double>::Random(10);
    cppoptlib::CMAesSolver<double> solver;
    solver.minimize(f, beta);

    return 0;
}
