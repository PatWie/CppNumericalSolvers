// CppNumericalSolver
#ifndef ARMIJO_H_
#define ARMIJO_H_

#include "../meta.h"

namespace cppoptlib {

template<typename T, typename P, int Ord>
class Armijo {

 public:

    /**
     * @brief use Armijo Rule for (weak) Wolfe conditiions
     * @details [long description]
     *
     * @param searchDir search direction for next update step
     * @param objFunc handle to problem
     *
     * @return step-width
     */
    static T linesearch(const Vector<T> & x, const Vector<T> & searchDir, P &objFunc, const  T alpha_init = 1.0) {
        const T c = 0.2;
        const T rho = 0.9;
        T alpha = alpha_init;
        T f = objFunc.value(x + alpha * searchDir);
        const T f_in = objFunc.value(x);
        Vector<T> grad(x.rows());
        objFunc.gradient(x, grad);
        const T Cache = c * grad.dot(searchDir);

        while(f > f_in + alpha * Cache) {
            alpha *= rho;
            f = objFunc.value(x + alpha * searchDir);
        }

        return alpha;
    }

};

template<typename T, typename P>
class Armijo<T, P, 2> {

 public:

    /**
     * @brief use Armijo Rule for (weak) Wolfe conditiions
     * @details [long description]
     *
     * @param searchDir search direction for next update step
     * @param objFunc handle to problem
     *
     * @return step-width
     */
    static T linesearch(const Vector<T> & x, const Vector<T> & searchDir, P &objFunc) {
        const T c = 0.2;
        const T rho = 0.9;
        T alpha = 1.0;

        T f = objFunc.value(x + alpha * searchDir);
        const T f_in = objFunc.value(x);
        const Matrix<T>  hessian(x.rows(), x.rows());
        objFunc.hessian(x, hessian);
        Vector<T> grad(x.rows());
        objFunc.gradient(x, grad);
        const T Cache = c * grad.dot(searchDir) + 0.5 * c*c * searchDir.transpose() * (hessian * searchDir);

        while(f > f_in + alpha * Cache) {
            alpha *= rho;
            f = objFunc.value(x + alpha * searchDir);
        }
        return alpha;
    }

};

}

#endif /* ARMIJO_H_ */
