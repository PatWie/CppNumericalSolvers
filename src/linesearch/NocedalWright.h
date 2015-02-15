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

#ifndef NOCEDALWRIGHT_H_
#define NOCEDALWRIGHT_H_


#include <functional>
#include <iostream>
#include "../Meta.h"

namespace pwie
{


// as in "Nocedal Wright Numerical Optimization"
class NocedalWright
{

public:

  static double zoom(const Vector &x, const Vector &p, double phi0_dash, double alpha_Lo, double alpha_Hi, double phi0, double phi_lo, function_t objective, gradient_t gradient)
  {
  	const double c1 = 0.0001;
    const double c2 = 0.9;
    const size_t DIM = x.size();

    double alpha;
    double phi = 0;

    // repeat
    for (int i = 0; i < 200; i++) {
    	// poor man's approach for interpolation
      alpha = (alpha_Lo + alpha_Hi) / 2;

      Vector x_test = x + alpha * p;
      phi = objective(x_test);

      // if phi(a_j) > phi(0) + c1*alpha_j*phi'(0) or phi(alpha_j) >= phi(alpha_lo)
      if ((phi > phi0 + c1 * alpha * phi0_dash) || (phi >= phi_lo)) {
        alpha_Hi = alpha;
      } else {
      	// evaluate phi'(alpha_j)
        Vector grad(DIM);
        gradient(x_test, grad);
        double phi_dash = grad.dot(p);

        // strong wolfe condition
        if (abs(phi_dash) <= -c2 * phi0_dash){
        	// stop
          return alpha;
        }

        // phi'(alpha_j)(alpha_hi - alpha_lo) >= 0
        if (phi_dash * (alpha_Hi - alpha_Lo) >= 0)
          alpha_Hi = alpha_Lo;

        alpha_Lo = alpha;
        phi_lo    = phi;
      }
    }
    return alpha;
  }

  static double linesearch(const Vector &x, const Vector &p, function_t objective, gradient_t gradient)
  {
    // constants
    const size_t DIM = x.size();
    const double c1 = 0.0001;
    const double c2 = 0.9;
    const double wAlphaMax = 100;

    double alpha = 1;
    double alpha_i_old = 0;
    double alpha_i = alpha;


    double phi0 = objective(x);

    Vector grad(DIM);
    Vector x_test(DIM);

    gradient(x, grad);

    // evaluate phi(alpha)
    double phi0_dash = grad.dot(p);

    size_t iter = 0;
    for(iter=0;;iter++) {
      x_test = x + alpha_i * p;
      double phi = objective(x_test);

      // decrease condition
      // phi(alpha_i) > phi(0)+c1*alpha_i*phi'(0) or [ phi(alpha_i) > phi(alpha_i-1) and i > 1]
      if ((phi > phi0 + c1 * alpha_i * phi0_dash) || ((phi >= phi0 ) && (iter > 0))) {
        return zoom(x, p, phi0_dash, alpha_i_old, alpha_i, phi0, phi0, objective, gradient);
      }

      // evaluate phi'(alpha_i)
      gradient(x_test, grad);
      double phi_dash = grad.dot(p);

      // strong curvature condition
      // | phi'(alpha_i) | <= -c_2 phi'(0)
      if (abs(phi_dash) <= -c2 * phi0_dash) {
        return alpha_i;
      }

      // phi(alpha_i) >= 0
      if (phi_dash >= 0) {
        return zoom(x, p, phi0_dash, alpha_i, alpha_i_old, phi0, phi, objective, gradient);
      }
      //std::cout << "alpha_i "<<alpha_i<<std::endl;
      alpha_i_old = alpha_i;
      alpha_i = min(wAlphaMax, 0.5*(alpha_i +wAlphaMax));
      phi0 = phi;
    }

    return alpha;
  }



};
}

#endif