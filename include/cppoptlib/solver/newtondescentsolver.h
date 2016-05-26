// CppNumericalSolver
#include <iostream>
#include <Eigen/LU>
#include "isolver.h"
#include "../linesearch/armijo.h"

#ifndef NEWTONDESCENTSOLVER_H_
#define NEWTONDESCENTSOLVER_H_

namespace cppoptlib {

template<typename T>
class NewtonDescentSolver : public ISolver<T, 2> {
  public:
    void minimize(Problem<T> &objFunc, Vector<T> & x0) {

        const size_t DIM = x0.rows();

        Vector<T> grad = Vector<T>::Zero(DIM);
        Matrix<T> hessian = Matrix<T>::Zero(DIM, DIM);

        T gradNorm = 0;

        this->m_current.reset();
        do {
            objFunc.gradient(x0, grad);
            objFunc.hessian(x0, hessian);
            hessian += (1e-5) * Matrix<T>::Identity(DIM, DIM);
            Vector<T> delta_x = hessian.lu().solve(-grad);

            const double rate = Armijo<T, decltype(objFunc), 1>::linesearch(x0, delta_x, objFunc) ;
            x0 = x0 + rate * delta_x;
            // std::cout << "iter: "<<iter<< ", f = " <<  objFunc.value(x0) << ", ||g||_inf "<<gradNorm  << std::endl;
            ++this->m_current.iterations;
            this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();
            this->m_status = checkConvergence(this->m_stop, this->m_current);
        } while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));
    }
};

}
/* namespace cppoptlib */

#endif /* NEWTONDESCENTSOLVER_H_ */
