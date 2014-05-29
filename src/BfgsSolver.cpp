/**
 * Copyright (c) 2014 Patrick Wieschollek
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

#include "BfgsSolver.h"
#include <iostream>
namespace pwie
{

BfgsSolver::BfgsSolver() : ISolver()
{

}


void BfgsSolver::internalSolve(Vector & x,
                               const FunctionOracleType & FunctionValue,
                               const GradientOracleType & FunctionGradient,
                               const HessianOracleType & FunctionHessian)
{


    const size_t DIM = x.rows();
    size_t iter = 0;
    Matrix H = Matrix::Identity(DIM, DIM);
    Vector grad(DIM);

    Vector x_old = x;

    do
    {
        FunctionGradient(x, grad);
        Vector p = -1 * H * grad;
        const double rate = linesearch(x, p, FunctionValue, FunctionGradient) ;
        x = x + rate * p;
        Vector grad_old = grad;
        FunctionGradient(x, grad);

        Vector s = x - x_old;
        Vector y = grad - grad_old;

        const double rho = 1.0 / y.dot(s);
        if(iter == 0)
        {
            H = ((y.dot(s)) / (y.dot(y)) * Matrix::Identity(DIM, DIM));
        }
        H = H - rho * (s * (y.transpose() * H) + (H * y) * s.transpose()) + rho * rho * (y.dot(H * y) + 1.0 / rho) * (s * s.transpose());
        x_old = x;
        iter++;

    }
    while((grad.lpNorm<Eigen::Infinity>() > settings.gradTol) && (iter < settings.maxIter));

}
}

/* namespace pwie */
