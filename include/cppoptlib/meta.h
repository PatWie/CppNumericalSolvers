// CppNumericalSolver
#ifndef META_H
#define META_H

#include <Eigen/Dense>

namespace cppoptlib {

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

typedef struct Options {
    double gradTol;
    double rate;
    size_t maxIter;
    size_t m;

    Options() {
        rate = 0.00005;
        maxIter = 100000;
        gradTol = 1e-5;
        m = 10;

    }
} Options;

template<typename T>
bool checkConvergence(T val_new, T val_old, Vector<T> grad, Vector<T> x_new, Vector<T> x_old) {

    T ftol = 1e-10;
    T gtol = 1e-8;
    T xtol = 1e-32;

    // value crit.
    if((x_new-x_old).cwiseAbs().maxCoeff() < xtol)
        return true;

    // // absol. crit
    if(abs(val_new - val_old) / (abs(val_new) + ftol) < ftol) {
        std::cout << abs(val_new - val_old) / (abs(val_new) + ftol) << std::endl;
        std::cout << val_new << std::endl;
        std::cout << val_old << std::endl;
        std::cout << abs(val_new - val_old) / (abs(val_new) + ftol) << std::endl;
        return true;
    }

    // gradient crit
    T g = grad.template lpNorm<Eigen::Infinity>();
    if (g < gtol)
        return true;
    return false;
}

}
#endif /* META_H */