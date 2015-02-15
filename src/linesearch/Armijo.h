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

#ifndef ARMIJO_H_
#define ARMIJO_H_


#include <functional>
#include "../Meta.h"

namespace pwie
{



class Armijo{

public:

	static double linesearch(const Vector & x, const Vector & p, function_t objective, gradient_t gradient){
		const double c = 0.2;
	    const double rho = 0.9;
	    double alpha = 1.0;

	    double f = objective(x + alpha * p);
	    const double f_in = objective(x);
	    Vector grad(x.rows());
	    gradient(x, grad);
	    const double Cache = c * grad.dot(p);

	    while(f > f_in + alpha * Cache)
	    {
	        alpha *= rho;
	        f = objective(x + alpha * p);
	    }

	    return alpha;
	}

};
}

#endif