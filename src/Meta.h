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


#ifndef META_H_
#define META_H_


#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdexcept>

namespace pwie
{
typedef std::function<double(const Eigen::VectorXd & x)> FunctionOracleType;
typedef std::function<void(const Eigen::VectorXd & x, Eigen::VectorXd & gradient)> GradientOracleType;
typedef std::function<void(const Eigen::VectorXd & x, Eigen::MatrixXd & hessian)> HessianOracleType;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::VectorXd::Scalar Scalar;

typedef struct Options
{
    double gradTol;
    double rate;
    size_t maxIter;
    int m;

    Options()
    {
        rate = 0.00005;
        maxIter = 100000;
        gradTol = 1e-5;
        m = 10;

    }
} Options;

bool checkGradient(const FunctionOracleType & FunctionValue, const Vector & x, const Vector & grad, const double eps = 1e-5);
void computeGradient(const FunctionOracleType & FunctionValue, const Vector & x, Vector & grad, const double eps = 1e-5);
void computeHessian(const FunctionOracleType & FunctionValue, const Vector & x, Matrix & hessian, const double eps = 1e-1);

const double EPS = 2.2204e-016;

template<typename T>
bool AssertSimiliar(T a, T b)
{
    return fabs(a - b) <=  1e-2;
}
template<typename T>
bool AssertGreaterThan(T a, T b)
{
    return (a - b) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * 1e-3);
}
template<typename T>
bool AssertLessThan(T a, T b)
{
    return (b - a) > ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * 1e-3);
}
template<typename T>
bool AssertEqual(T a, T b)
{
    return (a == b);
}
}

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define INF HUGE_VAL
#define Assert(x,m) if (!(x)) { throw (std::runtime_error(m)); }



#define FAST

#ifdef FAST
#define Debug(x)
#else
#define Debug(x) if(false){std::cout << "DEBUG: "<< x;std::cout<< std::endl;}
#endif

#endif /* META_H_ */
