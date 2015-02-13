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

#include "Meta.h"
#include <cmath>
#include <iostream>

namespace pwie
{

bool checkGradient(const FunctionOracleType & FunctionValue, const Vector & x, const Vector & grad, const double eps)
{
    const size_t DIM = x.rows();
    Vector finite(DIM);
    for(size_t i = 0; i < DIM; i++)
    {
        Vector xx = x;
        xx[i] += eps;
        Vector xy = x;
        xy[i] -= eps;
        finite[i] = (FunctionValue(xx) - FunctionValue(xy)) / (2.0 * eps);
    }
    const double error = static_cast<Vector>((finite - grad)).norm() / static_cast<Vector>((finite + grad)).norm();

    return !(error > eps);
}

void computeGradient(const FunctionOracleType & FunctionValue, const Vector & x, Vector & grad, const double eps)
{
    const size_t DIM = x.rows();
    Vector finite(DIM);
    for(size_t i = 0; i < DIM; i++)
    {
        Vector xx = x;
        xx[i] += eps;
        Vector xy = x;
        xy[i] -= eps;
        finite[i] = (FunctionValue(xx) - FunctionValue(xy)) / (2.0 * eps);
    }
    grad = finite;
}

void computeHessian(const FunctionOracleType & FunctionValue, const Vector & x, Matrix & hessian, const double eps)
{
    Assert(x.rows() == hessian.rows(), "hessian has wrong dimension (number of rows)");
    Assert(x.rows() == hessian.cols(), "hessian has wrong dimension (number of cols)");
    const size_t DIM = x.rows();
    for(size_t i = 0; i < DIM; i++)
    {
        for(size_t j = 0; j < DIM; j++)
        {
            Vector xx = x;
            double f4 = FunctionValue(xx);
            xx[i] += eps;
            xx[j] += eps;

            double f1 = FunctionValue(xx);

            xx[j] -= eps;
            double f2 = FunctionValue(xx);
            xx[j] += eps;
            xx[i] -= eps;
            double f3 = FunctionValue(xx);

            hessian(i, j) = (f1 - f2 - f3 + f4) / (eps * eps);
        }
    }
}



}

