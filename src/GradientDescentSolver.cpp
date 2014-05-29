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

#include "GradientDescentSolver.h"
#include <iostream>
namespace pwie
{

GradientDescentSolver::GradientDescentSolver() : ISolver()
{


}


void GradientDescentSolver::internalSolve(Vector & x,
        const FunctionOracleType & FunctionValue,
        const GradientOracleType & FunctionGradient,
        const HessianOracleType & FunctionHessian)
{
    Vector grad(x.rows());

    double x_old, x_new = FunctionValue(x);

    size_t iter = 0;
    do
    {
        x_old = x_new;
        FunctionGradient(x, grad);
        const double rate = linesearch(x, -grad, FunctionValue, FunctionGradient) ;

        x = x - rate * grad;
        x_new = FunctionValue(x);

        iter++;
    }
    while((grad.lpNorm<Eigen::Infinity>() > settings.gradTol) && (iter < settings.maxIter));


}
}

/* namespace pwie */
