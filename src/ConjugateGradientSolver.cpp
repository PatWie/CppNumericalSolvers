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

#include "ConjugateGradientSolver.h"
#include <iostream>
namespace pwie
{

ConjugateGradientSolver::ConjugateGradientSolver() : ISolver()
{


}


void ConjugateGradientSolver::internalSolve(Vector & x,
        const FunctionOracleType & FunctionValue,
        const GradientOracleType & FunctionGradient,
        const HessianOracleType & FunctionHessian)
{
    UNUSED(FunctionHessian);
    size_t iter = 0;

    Vector grad(x.rows());
    Vector grad_old(x.rows());
    Vector Si(x.rows());
    Vector Si_old(x.rows());
    do
    { 
        FunctionGradient(x, grad);

        if(iter==0){
            Si = -grad;
        }else{
            const double beta = grad.dot(grad)/(grad_old.dot(grad_old));
            Si = -grad + beta*Si_old;
        }
        
        const double rate = linesearch(x, Si, FunctionValue, FunctionGradient) ;

        x = x + rate * Si;

        iter++;
        grad_old = grad;
        Si_old = Si;
        
    }
    while((grad.lpNorm<Eigen::Infinity>() > settings.gradTol) && (iter < settings.maxIter));


}
}

/* namespace pwie */
