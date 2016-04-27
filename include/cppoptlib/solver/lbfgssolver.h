// CppNumericalSolver
#include <iostream>
#include <Eigen/LU>
#include "isolver.h"
#include "../linesearch/morethuente.h"

#ifndef LBFGSSOLVER_H_
#define LBFGSSOLVER_H_

namespace cppoptlib {

template<typename T>
class LbfgsSolver : public ISolver<T, 1> {
  public:
    void minimize(Problem<T> &objFunc, Vector<T> & x0) {

        const size_t m = 10;
        const size_t DIM = x0.rows();

        Matrix<T> sVector = Matrix<T>::Zero(DIM, m);
        Matrix<T> yVector = Matrix<T>::Zero(DIM, m);

        Vector<T> alpha = Vector<T>::Zero(m);
        Vector<T> grad(DIM), q(DIM), grad_old(DIM), s(DIM), y(DIM);
        objFunc.gradient(x0, grad);
        Vector<T> x_old = x0;
        Vector<T> x_old2 = x0;

        size_t iter = 0, globIter = 0;
        double H0k = 1;

        this->m_current.reset();
        do {
            const T relativeEpsilon = static_cast<T>(0.0001) * std::max(static_cast<T>(1.0), x0.norm());

            if (grad.norm() < relativeEpsilon)
                break;

            //Algorithm 7.4 (L-BFGS two-loop recursion)
            q = grad;
            const int k = std::min(m, iter);

            // for i = k − 1, k − 2, . . . , k − m
            for (int i = k - 1; i >= 0; i--) {
                // alpha_i <- rho_i*s_i^T*q
                const double rho = 1.0 / static_cast<Vector<T>>(sVector.col(i))
                .dot(static_cast<Vector<T>>(yVector.col(i)));
                alpha(i) = rho * static_cast<Vector<T>>(sVector.col(i)).dot(q);
                // q <- q - alpha_i*y_i
                q = q - alpha(i) * yVector.col(i);
            }
            // r <- H_k^0*q
            q = H0k * q;
            //for i k − m, k − m + 1, . . . , k − 1
            for (int i = 0; i < k; i++) {
                // beta <- rho_i * y_i^T * r
                const double rho = 1.0 / static_cast<Vector<T>>(sVector.col(i))
                .dot(static_cast<Vector<T>>(yVector.col(i)));
                const double beta = rho * static_cast<Vector<T>>(yVector.col(i)).dot(q);
                // r <- r + s_i * ( alpha_i - beta)
                q = q + sVector.col(i) * (alpha(i) - beta);
            }
            // stop with result "H_k*f_f'=q"

            // any issues with the descent direction ?
            double descent = -grad.dot(q);
            double alpha_init =  1.0 / grad.norm();
            if (descent > -0.0001 * relativeEpsilon) {
                q = -1 * grad;
                iter = 0;
                alpha_init = 1.0;
            }

            // find steplength
            const double rate = MoreThuente<T, decltype(objFunc), 1>::linesearch(x0, -q,  objFunc, alpha_init) ;
            // update guess
            x0 = x0 - rate * q;

            grad_old = grad;
            objFunc.gradient(x0, grad);

            s = x0 - x_old;
            y = grad - grad_old;

            // update the history
            if (iter < m) {
                sVector.col(iter) = s;
                yVector.col(iter) = y;
            } else {

                sVector.leftCols(m - 1) = sVector.rightCols(m - 1).eval();
                sVector.rightCols(1) = s;
                yVector.leftCols(m - 1) = yVector.rightCols(m - 1).eval();
                yVector.rightCols(1) = y;
            }
            // update the scaling factor
            H0k = y.dot(s) / static_cast<double>(y.dot(y));

            x_old = x0;
            // std::cout << "iter: "<<globIter<< ", f = " <<  objFunc.value(x0) << ", ||g||_inf "
            // <<gradNorm  << std::endl;

            iter++;
            globIter++;
            ++this->m_current.iterations;
            this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();
            this->m_status = checkConvergence(this->m_stop, this->m_current);
        } while (this->m_status == Status::Continue);

    }

};

}
/* namespace cppoptlib */

#endif /* LBFGSSOLVER_H_ */
