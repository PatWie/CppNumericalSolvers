// CppNumericalSolver
#include <iostream>
#include <Eigen/LU>
#include "isolver.h"
#include "../linesearch/armijo.h"

#ifndef BFGSSOLVER_H_
#define BFGSSOLVER_H_

namespace cns {

template<typename T>
class BfgsSolver : public ISolver<T, 1> {
  public:
    void minimize(Problem<T> &objFunc, Vector<T> & x0) {

        const size_t DIM = x0.rows();
        size_t iter = 0;
        Matrix<T> H = Matrix<T>::Identity(DIM, DIM);
        Vector<T> grad(DIM);
        T gradNorm = 0;
        Vector<T> x_old = x0;

        do {
            objFunc.gradient(x0, grad);
            Vector<T> searchDir = -1 * H * grad;
            // check "positive definite"
            double phi = grad.dot(searchDir);

            // positive definit ?
            if (phi > 0) {
                // no, we reset the hessian approximation
                H = Matrix<T>::Identity(DIM, DIM);
                searchDir = -1 * grad;
            }

            const double rate = Armijo<T, decltype(objFunc), 1>::linesearch(x0, searchDir, objFunc) ;
            x0 = x0 + rate * searchDir;

            Vector<T> grad_old = grad;
            objFunc.gradient(x0, grad);
            Vector<T> s = rate * searchDir;
            Vector<T> y = grad - grad_old;

            const double rho = 1.0 / y.dot(s);
            H = H - rho * (s * (y.transpose() * H) + (H * y) * s.transpose()) + rho * rho * (y.dot(H * y) + 1.0 / rho)
                * (s * s.transpose());
            gradNorm = grad.template lpNorm<Eigen::Infinity>();
            // std::cout << "iter: " << iter << ", f = " <<  objFunc.value(x0) << ", ||g||_inf " << gradNorm  << std::endl;

            x_old = x0;
            iter++;

        } while ((gradNorm > this->settings_.gradTol) && (iter < this->settings_.maxIter));

    }

};

}
/* namespace cns */

#endif /* BFGSSOLVER_H_ */