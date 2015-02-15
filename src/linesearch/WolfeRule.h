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

#ifndef WOLFERULE_H_
#define WOLFERULE_H_


#include <functional>
#include "../Meta.h"

namespace pwie
{


class WolfeRule {

public:

  static double linesearch(const Vector &x0, const Vector &z, function_t objective, gradient_t gradient) {

    double alpha = 1.0;
    Vector x = x0;

    // evaluate phi(0)
    double phi0 = objective(x0);

    // evaluate phi'(0)
    Vector grad(x.rows());
    gradient(x, grad);
    double phi0_dash = z.dot(grad);



    bool decrease_direction = true;

    // 200 guesses
    for (size_t iter = 0; iter < 200; ++iter) {

      // new guess for phi(alpha)
      Vector x_candidate = x + alpha * z;
      const double phi = objective(x_candidate);

      // decrease condition invalid --> shrink interval
      if (phi > phi0 + 0.0001 * alpha * phi0_dash) {
        alpha *= 0.5;
        decrease_direction = false;
      }
      else {

        // valid decrease --> test strong wolfe condition
        Vector grad2(x.rows());
        gradient(x_candidate, grad2);
        const double phi_dash = z.dot(grad2);

        // curvature condition invalid ?
        if ((phi_dash < 0.9 * phi0_dash) || !decrease_direction) { 
        	// increase interval
          alpha *= 4.0;
        }
        else {
        	// both condition are valid --> we are happy
          x = x_candidate;
          grad = grad2;
          phi0 = phi;
          return alpha;
        }
      }
    }


    return alpha;
  }

};
}

#endif