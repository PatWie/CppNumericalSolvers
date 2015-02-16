
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

#include "LbfgsSolver.h"
#include "linesearch/WolfeRule.h"
#include <iostream>
namespace pwie
{

LbfgsSolver::LbfgsSolver() : ISolver()
{
  // TODO Auto-generated constructor stub

}


void LbfgsSolver::internalSolve(Vector &x,
                                const function_t &FunctionValue,
                                const gradient_t &FunctionGradient,
                                const hessian_t &FunctionHessian)
{
  UNUSED(FunctionHessian);
  const size_t m = 10;
  const size_t DIM = x.rows();


  Matrix sVector = Matrix::Zero(DIM, m);
  Matrix yVector = Matrix::Zero(DIM, m);

  Vector alpha = Vector::Zero(m);
  Vector grad(DIM), q(DIM), grad_old(DIM), s(DIM), y(DIM);
  FunctionGradient(x, grad);
  Vector x_old = x;
  Vector x_old2 = x;

  double x_start = FunctionValue(x), x_end;

  size_t iter = 0, j=0;

  double H0k = 1;

  
  double oscillate_diff=0,oscillate_diff2=0;

  do
  {

    const double relativeEpsilon = 0.0001 * max(1.0, x.norm());

    if (grad.norm() < relativeEpsilon)
      break;

    //Algorithm 7.4 (L-BFGS two-loop recursion)
    q = grad;
    const int k = min(m, iter);

    // for i k − 1, k − 2, . . . , k − m
    for (int i = k - 1; i >= 0; i--)
    {
      // alpha_i <- rho_i*s_i^T*q
      const double rho = 1.0 / static_cast<Vector>(sVector.col(i)).dot(static_cast<Vector>(yVector.col(i)));
      alpha(i) = rho * static_cast<Vector>(sVector.col(i)).dot(q);
      // q <- q - alpha_i*y_i
      q = q - alpha(i) * yVector.col(i);
    }
    // r <- H_k^0*q
    q = H0k * q;
    //for i k − m, k − m + 1, . . . , k − 1
    for (int i = 0; i < k; i++)
    {
      // beta <- rho_i * y_i^T * r
      const double rho = 1.0 / static_cast<Vector>(sVector.col(i)).dot(static_cast<Vector>(yVector.col(i)));
      const double beta = rho * static_cast<Vector>(yVector.col(i)).dot(q);
      // r <- r + s_i * ( alpha_i - beta)
      q = q + sVector.col(i) * (alpha(i) - beta);
    }
    // stop with result "H_k*f_f'=q"

    // any issues with the descent direction ?
    double descent = -grad.dot(q);
    double alpha_init =  1.0/grad.norm();
    if (descent > -0.0001 * relativeEpsilon) {
       q = -1*grad;
       iter = 0;
       alpha_init = 1.0;
    }

    // find steplength
    const double rate = WolfeRule::linesearch(x, -q,  FunctionValue, FunctionGradient, alpha_init) ;
    // update guess
    x = x - rate * q;

    grad_old = grad;
    FunctionGradient(x, grad);

    s = x - x_old;
    y = grad - grad_old;

    // update the history
    if (iter < m)
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
    // update the scaling factor
    H0k = y.dot(s) / static_cast<double>(y.dot(y));


    // now the ugly part : detect convergence
    // observation: L-BFGS seems to oscillate
    x_start = FunctionValue(x_old);
    x_old2 = x_old;
    x_old = x;
    x_end = FunctionValue(x);

    oscillate_diff2 = oscillate_diff;
    oscillate_diff = static_cast<Vector>(x_old2-x).norm();

    iter++;
    j++;

    if(fabs(oscillate_diff-oscillate_diff2)<1.0e-7)
      break;


  }
  while ((grad.norm() > 1.0e-5) && (j < settings.maxIter));



}
}

/* namespace pwie */
