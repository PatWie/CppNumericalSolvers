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

#ifndef LBFGSBSOLVER_H_
#define LBFGSBSOLVER_H_
#include "ISolver.h"
#include <list>
namespace pwie
{

/* coded from scratch !!!
 * based on the paper
 * A LIMITED MEMORY ALGORITHM FOR BOUND CONSTRAINED OPTIMIZATION
 * (Byrd, Lu, Nocedal, Zhu)
 */
class LbfgsbSolver : public ISolver
{
    Vector lb;
    Vector ub;
    std::list<Vector> xHistory;

    FunctionOracleType FunctionObjectiveOracle_;
    GradientOracleType FunctionGradientOracle_;

    Matrix W, M;
    double theta;
    int DIM;
    bool hasbounds = false;
private:
    /// <summary>
    /// find cauchy point in x
    /// </summary>
    /// <parameter name="x">start in x</parameter>
    void GetGeneralizedCauchyPoint(Vector & x, Vector & g, Vector & x_cauchy, Vector & c);
    /// <summary>
    /// find valid alpha for (8.5)
    /// </summary>
    /// <parameter name="x_cp">cauchy point</parameter>
    /// <parameter name="du">unconstrained solution of subspace minimization</parameter>
    /// <parameter name="FreeVariables">flag (1 if is free variable and 0 if is not free variable)</parameter>
    double FindAlpha(Vector & x_cp, Vector & du, std::vector<int> & FreeVariables);
    /// <summary>
    /// using linesearch to determine step width
    /// </summary>
    /// <parameter name="x">start in x</parameter>
    /// <parameter name="dx">direction</parameter>
    /// <parameter name="f">current value of objective (will be changed)</parameter>
    /// <parameter name="g">current gradient of objective (will be changed)</parameter>
    /// <parameter name="t">step width (will be changed)</parameter>
    void LineSearch(Vector & x, Vector dx, double & f, Vector & g, double & t) ;
    /// <summary>
    /// direct primal approach
    /// </summary>
    /// <parameter name="x">start in x</parameter>
    void SubspaceMinimization(Vector & x_cauchy, Vector & x, Vector & c, Vector & g, Vector & SubspaceMin);
public:
    LbfgsbSolver();
    void internalSolve(Vector & x0,
                       const FunctionOracleType & FunctionValue,
                       const GradientOracleType & FunctionGradient,
                       const HessianOracleType & FunctionHessian = std::function<void(const Eigen::VectorXd & x, Eigen::MatrixXd & hessian)>());

    void setBounds(const Vector & lower, const Vector & upper);
};

} /* namespace pwie */

#endif /* LBFGSBSOLVER_H_ */
