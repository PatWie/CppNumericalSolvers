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

#include "LbfgsSolver.h"
#include <iostream>
namespace pwie
{

LbfgsSolver::LbfgsSolver() : ISolver()
{
    // TODO Auto-generated constructor stub

}


void LbfgsSolver::internalSolve(Vector & x,
                                const FunctionOracleType & FunctionValue,
                                const GradientOracleType & FunctionGradient,
                                const HessianOracleType & FunctionHessian)
{
    UNUSED(FunctionHessian);
    const size_t m = 10;
    const size_t DIM = x.rows();


    Matrix sVector = Matrix::Zero(DIM, m);
    Matrix yVector = Matrix::Zero(DIM, m);
    Matrix H = Matrix::Identity(DIM, DIM);

    Vector alpha = Vector::Zero(m);
    Vector grad(DIM);
    FunctionGradient(x, grad);
    Vector x_old = x;

    size_t iter = 0;

    do
    {
        Vector q = grad;
        const int mini = min(m, iter);

        for(int i = mini - 1; i >= 0; i--)
        {
            const double rho = 1.0 / static_cast<Vector>(sVector.col(i)).dot(static_cast<Vector>(yVector.col(i)));
            alpha(i) = rho * static_cast<Vector>(sVector.col(i)).dot(q);
            q = q - alpha(i) * yVector.col(i);
        }
        q = H * q;
        for(int i = 0; i < mini; i++)
        {
            const double rho = 1.0 / static_cast<Vector>(sVector.col(i)).dot(static_cast<Vector>(yVector.col(i)));
            double beta = rho * static_cast<Vector>(yVector.col(i)).dot(q);
            q = q + sVector.col(i) * (alpha(i) - beta);
        }

        const double rate = linesearch(x, -q, H, FunctionValue, FunctionGradient) ;
        x = x - rate * q;
        Vector grad_old = grad;
        FunctionGradient(x, grad);

        Vector s = x - x_old;
        Vector y = grad - grad_old;

        if(iter < m)
        {
            sVector.col(iter) = s;
            yVector.col(iter) = y;
        }
        else
        {

            sVector.leftCols(m - 1) = sVector.rightCols(m - 1).eval();
            sVector.rightCols(1) = s;
            yVector.leftCols(m - 1) = yVector.rightCols(m - 1).eval();
            yVector.rightCols(1) = y;
        }

        H = y.dot(s) / static_cast<double>(y.dot(y)) * Matrix::Identity(DIM, DIM);
        x_old = x;

        //std::cout << FunctionValue(x) << std::endl;
        iter++;

    }
    while((grad.lpNorm<Eigen::Infinity>() > settings.gradTol) && (iter < settings.maxIter));



}
}

/* namespace pwie */
