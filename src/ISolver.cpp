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

#include "ISolver.h"

namespace pwie
{

ISolver::ISolver()
{
    settings = Options();

}

ISolver::~ISolver()
{
    // TODO Auto-generated destructor stub
}

void ISolver::solve(Eigen::VectorXd & x0,
                    const FunctionOracleType & FunctionValue,
                    const GradientOracleType & FunctionGradient,
                    const HessianOracleType & FunctionHessian)
{

    auto derivative = FunctionGradient;
    if(!derivative)
    {
        derivative = [&](const Vector x, Vector & grad) -> void
        {
            grad = Vector::Zero(x.rows());
            computeGradient(FunctionValue, x , grad);
        };
    }
    internalSolve(x0, FunctionValue, derivative, FunctionHessian);

}

double ISolver::linesearch(const Vector & x, const Vector & direction,
                           const FunctionOracleType & FunctionValue,
                           const GradientOracleType & FunctionGradient)
{

    const double alpha = 0.2;
    const double beta = 0.9;
    double t = 1.0;

    double f = FunctionValue(x + t * direction);
    const double f_in = FunctionValue(x);
    Vector grad(x.rows());
    FunctionGradient(x, grad);
    const double Cache = alpha * grad.dot(direction);

    while(f > f_in + t * Cache)
    {
        t *= beta;
        f = FunctionValue(x + t * direction);
    }

    return t;

}

} /* namespace pwie */
