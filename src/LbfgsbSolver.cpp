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

#include "LbfgsbSolver.h"
#include <iostream>
#include <vector>
#include <list>

namespace pwie
{

std::vector<int> sort_indexes(const std::vector< std::pair<int, double> > & v)
{
    std::vector<int> idx(v.size());
    for(size_t i = 0; i != idx.size(); ++i)
        idx[i] = v[i].first;
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2)
    {
        return v[i1].second < v[i2].second;
    });
    return idx;
}


LbfgsbSolver::LbfgsbSolver() : ISolver()
{
    // TODO Auto-generated constructor stub
    hasbounds = false;

}

void LbfgsbSolver::setBounds(const Vector & lower, const Vector & upper)
{
    lb = lower;
    ub = upper;
    hasbounds = true;
}

void LbfgsbSolver::GetGeneralizedCauchyPoint(Vector & x, Vector & g, Vector & x_cauchy,
        Vector & c)
{
    const int DIM = x.rows();
    // PAGE 8
    // Algorithm CP: Computation of the generalized Cauchy point
    // Given x,l,u,g, and B = \theta I-WMW

    // {all t_i} = { (idx,value), ... }
    // TODO: use "std::set" ?
    std::vector<std::pair<int, double> > SetOfT;
    // the feasible set is implicitly given by "SetOfT - {t_i==0}"
    Vector d = Vector::Zero(DIM, 1);

    // n operations
    for(int j = 0; j < DIM; j++)
    {
        if(g(j) == 0)
        {
            SetOfT.push_back(std::make_pair(j, INF));
        }
        else
        {
            double tmp = 0;
            if(g(j) < 0)
            {
                tmp = (x(j) - ub(j)) / g(j);
            }
            else
            {
                tmp = (x(j) - lb(j)) / g(j);
            }
            d(j) = -g(j);
            SetOfT.push_back(std::make_pair(j, tmp));
        }

    }
    Debug(d.transpose());

    // paper: using heapsort
    // sortedindices [1,0,2] means the minimal element is on the 1th entry
    std::vector<int> SortedIndices = sort_indexes(SetOfT);

    x_cauchy = x;
    // Initialize
    // p :=     W^T*p
    Vector p = (W.transpose() * d);                     // (2mn operations)
    // c :=     0
    c = Eigen::MatrixXd::Zero(M.rows(), 1);
    // f' :=    g^T*d = -d^Td
    double f_prime = -d.dot(d);                         // (n operations)
    // f'' :=   \theta*d^T*d-d^T*W*M*W^T*d = -\theta*f' - p^T*M*p
    double f_doubleprime = (double)(-1.0 * theta) * f_prime - p.dot(M * p); // (O(m^2) operations)
    // \delta t_min :=  -f'/f''
    double dt_min = -f_prime / f_doubleprime;
    // t_old :=     0
    double t_old = 0;
    // b :=     argmin {t_i , t_i >0}
    int i = 0;
    for(int j = 0; j < DIM; j++)
    {
        i = j;
        if(SetOfT[SortedIndices[j]].second != 0)
            break;
    }
    int b = SortedIndices[i];
    // see below
    // t                    :=  min{t_i : i in F}
    double t = SetOfT[b].second;
    // \delta t             :=  t - 0
    double dt = t - t_old;

    // examination of subsequent segments
    while((dt_min >= dt) && (i < DIM))
    {
        if(d(b) > 0)
            x_cauchy(b) = ub(b);
        else if(d(b) < 0)
            x_cauchy(b) = lb(b);

        // z_b = x_p^{cp} - x_b
        double zb = x_cauchy(b) - x(b);
        // c   :=  c +\delta t*p
        c += dt * p;
        // cache
        Vector wbt = W.row(b);

        f_prime += dt * f_doubleprime + (double) g(b) * g(b)
                   + (double) theta * g(b) * zb
                   - (double) g(b) * wbt.transpose() * (M * c);
        f_doubleprime += (double) - 1.0 * theta * g(b) * g(b)
                         - (double) 2.0 * (g(b) * (wbt.dot(M * p)))
                         - (double) g(b) * g(b) * wbt.transpose() * (M * wbt);
        p += g(b) * wbt.transpose();
        d(b) = 0;
        dt_min = -f_prime / f_doubleprime;
        t_old = t;
        ++i;
        if(i < DIM)
        {
            b = SortedIndices[i];
            t = SetOfT[b].second;
            dt = t - t_old;
        }

    }

    dt_min = max(dt_min, 0);
    t_old += dt_min;

    Debug(SortedIndices[0] << " " << SortedIndices[1]);

    #pragma omp parallel for
    for(int ii = i; ii < x_cauchy.rows(); ii++)
    {
        x_cauchy(SortedIndices[ii]) = x(SortedIndices[ii])
                                      + t_old * d(SortedIndices[ii]);
    }
    Debug(x_cauchy.transpose());

    c += dt_min * p;
    Debug(c.transpose());

}
double LbfgsbSolver::FindAlpha(Vector & x_cp, Vector & du, std::vector<int> & FreeVariables)
{
    /* this returns
     * a* = max {a : a <= 1 and  l_i-xc_i <= a*d_i <= u_i-xc_i}
     */
    double alphastar = 1;
    const unsigned int n = FreeVariables.size();
    for(unsigned int i = 0; i < n; i++)
    {
        if(du(i) > 0)
        {
            alphastar = min(alphastar,
                            (ub(FreeVariables[i]) - x_cp(FreeVariables[i]))
                            / du(i));
        }
        else
        {
            alphastar = min(alphastar,
                            (lb(FreeVariables[i]) - x_cp(FreeVariables[i]))
                            / du(i));
        }
    }
    return alphastar;
}

void LbfgsbSolver::LineSearch(Vector & x, Vector dx, double & f, Vector & g, double & t)
{

    const double alpha = 0.2;
    const double beta = 0.8;

    const double f_in = f;
    const Vector g_in = g;
    const double Cache = alpha * g_in.dot(dx);

    t = 1.0;
    f = FunctionObjectiveOracle_(x + t * dx);
    while(f > f_in + t * Cache)
    {
        t *= beta;
        f = FunctionObjectiveOracle_(x + t * dx);
    }
    FunctionGradientOracle_(x + t * dx, g);
    x += t * dx;

}

void LbfgsbSolver::SubspaceMinimization(Vector & x_cauchy, Vector & x, Vector & c, Vector & g,
                                        Vector & SubspaceMin)
{

    // cached value: ThetaInverse=1/theta;
    double theta_inverse = 1 / theta;

    // size of "t"
    std::vector<int> FreeVariablesIndex;
    Debug(x_cauchy.transpose());

    //std::cout << "free vars " << FreeVariables.rows() << std::endl;
    for(int i = 0; i < x_cauchy.rows(); i++)
    {
        Debug(x_cauchy(i) << " " << ub(i) << " " << lb(i));
        if((x_cauchy(i) != ub(i)) && (x_cauchy(i) != lb(i)))
        {
            FreeVariablesIndex.push_back(i);
        }
    }
    const int FreeVarCount = FreeVariablesIndex.size();

    Matrix WZ = Matrix::Zero(W.cols(), FreeVarCount);

    for(int i = 0; i < FreeVarCount; i++)
        WZ.col(i) = W.row(FreeVariablesIndex[i]);

    Debug(WZ);

    // r=(g+theta*(x_cauchy-x)-W*(M*c));
    Debug(g);
    Debug(x_cauchy);
    Debug(x);
    Vector rr = (g + theta * (x_cauchy - x) - W * (M * c));
    // r=r(FreeVariables);
    Vector r = Matrix::Zero(FreeVarCount, 1);
    for(int i = 0; i < FreeVarCount; i++)
        r.row(i) = rr.row(FreeVariablesIndex[i]);

    Debug(r.transpose());

    // STEP 2: "v = w^T*Z*r" and STEP 3: "v = M*v"
    Vector v = M * (WZ * r);
    // STEP 4: N = 1/theta*W^T*Z*(W^T*Z)^T
    Matrix N = theta_inverse * WZ * WZ.transpose();
    // N = I - MN
    N = Matrix::Identity(N.rows(), N.rows()) - M * N;
    // STEP: 5
    // v = N^{-1}*v
    v = N.lu().solve(v);
    // STEP: 6
    // HERE IS A MISTAKE IN THE ORIGINAL PAPER!
    Vector du = -theta_inverse * r
                - theta_inverse * theta_inverse * WZ.transpose() * v;
    Debug(du.transpose());
    // STEP: 7
    double alpha_star = FindAlpha(x_cauchy, du, FreeVariablesIndex);

    // STEP: 8
    Vector dStar = alpha_star * du;

    SubspaceMin = x_cauchy;
    for(int i = 0; i < FreeVarCount; i++)
    {
        SubspaceMin(FreeVariablesIndex[i]) = SubspaceMin(
                FreeVariablesIndex[i]) + dStar(i);
    }
}

void LbfgsbSolver::internalSolve(Vector & x0,
                                 const FunctionOracleType & FunctionValue,
                                 const GradientOracleType & FunctionGradient,
                                 const HessianOracleType & FunctionHessian)
{

    DIM = x0.rows();
    if(!hasbounds)
    {
        lb = (-1 * Vector::Ones(DIM)) * INF;
        ub = Vector::Ones(DIM) * INF;
        hasbounds = true;
    }
    theta = 1.0;

    W = Matrix::Zero(DIM, 0);
    M = Matrix::Zero(0, 0);


    FunctionObjectiveOracle_ = FunctionValue;
    FunctionGradientOracle_ = FunctionGradient;

    Assert(x0.rows() == lb.rows(), "lower bound size incorrect");
    Assert(x0.rows() == ub.rows(), "upper bound size incorrect");

    Debug(x0.transpose());
    Debug(lb.transpose());
    Debug(ub.transpose());

    Assert((x0.array() >= lb.array()).all(),
           "seed is not feasible (violates lower bound)");
    Assert((x0.array() <= ub.array()).all(),
           "seed is not feasible (violates upper bound)");



    xHistory.push_back(x0);

    Matrix yHistory = Matrix::Zero(DIM, 0);
    Matrix sHistory = Matrix::Zero(DIM, 0);

    Vector x = x0, g;
    int k = 0;

    double f = FunctionObjectiveOracle_(x);

    FunctionGradientOracle_(x, g);

    Debug(f);
    Debug(g.transpose());



    auto noConvergence =
        [&](Vector & x, Vector & g)->bool
    {
        return (((x - g).cwiseMax(lb).cwiseMin(ub) - x).lpNorm<Eigen::Infinity>() >= 1e-4);
    };

    while(noConvergence(x, g) && (k < settings.maxIter))
    {

        //std::cout << FunctionObjectiveOracle_(x) << std::endl;
        Debug("iteration " << k)
        double f_old = f;
        Vector x_old = x;
        Vector g_old = g;

        // STEP 2: compute the cauchy point by algorithm CP
        Vector CauchyPoint = Matrix::Zero(DIM, 1), c = Matrix::Zero(DIM, 1);
        GetGeneralizedCauchyPoint(x, g, CauchyPoint, c);
        // STEP 3: compute a search direction d_k by the primal method
        Vector SubspaceMin;
        SubspaceMinimization(CauchyPoint, x, c, g, SubspaceMin);

        Matrix H;
        double Length = 0;

        // STEP 4: perform linesearch and STEP 5: compute gradient
        LineSearch(x, SubspaceMin - x, f, g, Length);

        xHistory.push_back(x);

        // prepare for next iteration
        Vector newY = g - g_old;
        Vector newS = x - x_old;

        // STEP 6:
        double test = newS.dot(newY);
        test = (test < 0) ? -1.0 * test : test;

        if(test > EPS * newY.squaredNorm())
        {
            if(k < settings.m)
            {
                yHistory.conservativeResize(DIM, k + 1);
                sHistory.conservativeResize(DIM, k + 1);
            }
            else
            {

                yHistory.leftCols(settings.m - 1) = yHistory.rightCols(
                                                        settings.m - 1).eval();
                sHistory.leftCols(settings.m - 1) = sHistory.rightCols(
                                                        settings.m - 1).eval();
            }
            yHistory.rightCols(1) = newY;
            sHistory.rightCols(1) = newS;

            // STEP 7:
            theta = (double)(newY.transpose() * newY)
                    / (newY.transpose() * newS);

            W = Matrix::Zero(yHistory.rows(),
                             yHistory.cols() + sHistory.cols());

            W << yHistory, (theta * sHistory);

            Matrix A = sHistory.transpose() * yHistory;
            Matrix L = A.triangularView<Eigen::StrictlyLower>();
            Matrix MM(A.rows() + L.rows(), A.rows() + L.cols());
            Matrix D = -1 * A.diagonal().asDiagonal();
            MM << D, L.transpose(), L, ((sHistory.transpose() * sHistory)
                                        * theta);

            M = MM.inverse();
        }

        Vector ttt = Matrix::Zero(1, 1);
        ttt(0) = f_old - f;
        Debug("--> " << ttt.norm());
        if(ttt.norm() < 1e-8)
        {
            // successive function values too similar
            break;
        }
        k++;

    }


    x0 = x;



}
}

/* namespace pwie */
