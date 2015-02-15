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
                    const function_t & FunctionValue,
                    const gradient_t & FunctionGradient,
                    const hessian_t & FunctionHessian)
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
                           const function_t & FunctionValue,
                           const gradient_t & FunctionGradient)
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
/*
function alphas = strongwolfe(f,d,x0,alpham)
% function alphas = strongwolfe(f,d,x0,alpham)
% Line search algorithm satisfying strong Wolfe conditions
% Algorithms 3.5 on pages 60-61 in Nocedal and Wright
% MATLAB code by Kartik Sivaramakrishnan
% Last modified: January 27, 2008

alpha0 = 0;
alphap = alpha0;
c1 = 1e-4;
c2 = 0.5;
alphax = alpham*rand(1);
[fx0,gx0] = feval(f,x0,d);
fxp = fx0;
gxp = gx0;
i=1;
% alphap is alpha_{i-1}
% alphax is alpha_i
while (1 ~= 2)
  xx = x0 + alphax*d;
  [fxx,gxx] = feval(f,xx,d);
  if (fxx > fx0 + c1*alphax*gx0) | ((i > 1) & (fxx >= fxp)),
    alphas = zoom(f,x0,d,alphap,alphax);
    return;
  end
  if abs(gxx) <= -c2*gx0,
    alphas = alphax;
    return;
  end
  if gxx >= 0,
    alphas = zoom(f,x0,d,alphax,alphap);
    return;
  end
  alphap = alphax;
  fxp = fxx;
  gxp = gxx;
  alphax = alphax + (alpham-alphax)*rand(1);
  i = i+1;
end
*/
/*double ISolver::linesearch(const Vector & x, const Vector & d,
                           const function_t & FunctionValue,
                           const gradient_t & FunctionGradient){
    double alpha0 = 0;
    double alphap = alpha0;
    double c1 = 1e-4;
    double c2 = 0.5;
    double alphax = 1;

    double fx0 = FunctionValue(x);
    Vector gx0(x.rows());
    FunctionGradient(x, gx0);

    double fxp = fx0;
    Vector gxp = gx0;

    int i = 1;

    while(true){
        Vector xx = x0 + alphax*d;
        double fxx = FunctionValue(xx);

        Vector gxx(x.rows());
        FunctionGradient(xx, gxx);

        if (fxx > fx0 + c1*alphax*gx0) | ((i > 1) & (fxx >= fxp)){
            return zoom(FunctionValue, FunctionGradient, x0, d, alphap, alphax);
        }
        if( gxx.cwiseAbs() <= -c2*gx0 ){
            return alphax;
        }
        if(gxx >= 0){
            return zoom(FunctionValue, FunctionGradient, x0, d, alphax, alphap);
        }
        alphap = alphax;
        fxp = fxx;
        gxp = gxx;
        alphax = alphax + (alpham-alphax);
        i++;
    }

}*/


double ISolver::linesearch(const Vector & x, const Vector & direction,
                         const Eigen::MatrixXd & hessian,
                           const function_t & FunctionValue,
                           const gradient_t & FunctionGradient)
{

    const double alpha = 0.2;
    const double beta = 0.9;
    double t = 1.0;

    double f = FunctionValue(x + t * direction);
    const double f_in = FunctionValue(x);
    Vector grad(x.rows());
    FunctionGradient(x, grad);
    const double Cache = alpha * grad.dot(direction) + 0.5*alpha*direction.transpose()*(hessian*direction);

    while(f > f_in + t * Cache)
    {
        t *= beta;
        f = FunctionValue(x + t * direction);
    }

    return t;

}

} /* namespace pwie */
