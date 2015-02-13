/**
 * Copyright (c) 2014-2015 Patrick Wieschollek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "NewtonDescentSolver.h"
#include <iostream>
#include <Eigen/LU>
namespace pwie
{

NewtonDescentSolver::NewtonDescentSolver() : ISolver()
{

}


void NewtonDescentSolver::internalSolve(Vector & x,
                                        const FunctionOracleType & FunctionValue,
                                        const GradientOracleType & FunctionGradient,
                                        const HessianOracleType & FunctionHessian)
{

    if(!FunctionHessian)
    {
        std::cout << "Error: hessian matrix is missing!" << std::endl;
        return;
    }

    const size_t DIM = x.rows();

    Vector grad = Vector::Zero(DIM);
    Matrix hessian = Matrix::Zero(DIM, DIM);

    FunctionGradient(x, grad);
    FunctionHessian(x, hessian);

    size_t iter = 0;
    do
    {
        hessian += (1e-5) * Matrix::Identity(DIM, DIM);
        Vector delta_x = hessian.lu().solve(-grad);
        const double rate = linesearch(x, delta_x, FunctionValue, FunctionGradient) ;
        x = x + rate * delta_x;

        FunctionGradient(x, grad);
        FunctionHessian(x, hessian);

        iter++;
    }
    while((grad.lpNorm<Eigen::Infinity>() > settings.gradTol) && (iter < settings.maxIter));


}
}

/* namespace pwie */
