/*
 * unittest.hpp
 *      Author: Patrick Wieschollek
 */

#ifndef UNITTEST_HPP_
#define UNITTEST_HPP_

#include <iostream>
#include <functional>
#include <list>
#include "Meta.h"
#include "BfgsSolver.h"
#include "LbfgsSolver.h"
#include "LbfgsbSolver.h"
#include "GradientDescentSolver.h"
#include "NewtonDescentSolver.h"

using namespace pwie;


void runtest(void)
{

    std::cout  << std::endl;
    // create function
    auto rosenbrock = [](const Vector & x) -> double
    {
        const double t1 = (1 - x[0]);
        const double t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    };

    // create derivative of function
    auto Drosenbrock = [](const Vector x, Vector & grad) -> void
    {
        grad = Vector::Zero(2);
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1]  = 200 * (x[1] - x[0] * x[0]);
    };
    // create hessian of function (numerical approximation)
    auto DDrosenbrock = [&](const Vector x, Matrix & hes) -> void
    {
        hes = Matrix::Zero(x.rows(), x.rows());
        computeHessian(rosenbrock, x, hes);
    };

    std::list<std::pair<double, double>> test = {{15.0, 8.0}, {1.0, 3.0}};

    for(auto i : test)
    {
        // initial guess
        Vector x0(2);
        x0(0) = i.first;
        x0(1) = i.second;
        std::cout << std::endl;
        std::cout << "test rosenbrock function starting in ";
        std::cout << x0.transpose();
        std::cout << std::endl;

        // get derivative
        Vector dx0(2);
        Drosenbrock(x0, dx0);
        // verify gradient by finite differences
        checkGradient(rosenbrock, x0, dx0);

        // use solver (GradientDescentSolver,BfgsSolver,LbfgsSolver,LbfgsbSolver)
        Vector x = x0;
        BfgsSolver bfgs;
        bfgs.solve(x, rosenbrock, Drosenbrock);
        if(!AssertSimiliar(rosenbrock(x), 0.0))
            std::cout << "[FAILED] BfgsSolver with error " << rosenbrock(x) << std::endl;
        else
            std::cout << "[OK] BfgsSolver " << std::endl;

        x = x0;
        bfgs.solve(x, rosenbrock);
        if(!AssertSimiliar(rosenbrock(x), 0.0))
            std::cout << "[FAILED] BfgsSolver (auto derivative) with error " << rosenbrock(x) << std::endl;
        else
            std::cout << "[OK] BfgsSolver (auto derivative) " << std::endl;

        x = x0;
        LbfgsSolver lbfgs;
        lbfgs.solve(x, rosenbrock, Drosenbrock);
        if(!AssertSimiliar(rosenbrock(x), 0.0))
            std::cout << "[FAILED] LbfgsSolver with error " << rosenbrock(x) << std::endl;
        else
            std::cout << "[OK] LbfgsSolver " << std::endl;

        x = x0;

        lbfgs.solve(x, rosenbrock);
        if(!AssertSimiliar(rosenbrock(x), 0.0))
            std::cout << "[FAILED] LbfgsSolver (auto derivative) with error " << rosenbrock(x) << std::endl;
        else
            std::cout << "[OK] LbfgsSolver (auto derivative) " << std::endl;

        x = x0;
        GradientDescentSolver gradesc;
        gradesc.solve(x, rosenbrock, Drosenbrock);
        if(!AssertSimiliar(rosenbrock(x), 0.0))
            std::cout << "[FAILED] GradientDescentSolver with error " << rosenbrock(x) << std::endl;
        else
            std::cout << "[OK] GradientDescentSolver " << std::endl;

        x = x0;

        gradesc.solve(x, rosenbrock);
        if(!AssertSimiliar(rosenbrock(x), 0.0))
            std::cout << "[FAILED] GradientDescentSolver (auto derivative) with error " << rosenbrock(x) << std::endl;
        else
            std::cout << "[OK] GradientDescentSolver (auto derivative) " << std::endl;

        x = x0;
        NewtonDescentSolver newtondesc;
        newtondesc.solve(x, rosenbrock, Drosenbrock, DDrosenbrock);
        if(!AssertSimiliar(rosenbrock(x), 0.0))
            std::cout << "[FAILED] NewtonDescentSolver with error " << rosenbrock(x) << std::endl;
        else
            std::cout << "[OK] NewtonDescentSolver " << std::endl;

        x = x0;
        LbfgsbSolver lbfgsb;
        lbfgsb.solve(x, rosenbrock, Drosenbrock);
        if(!AssertSimiliar(rosenbrock(x), 0.0))
            std::cout << "[FAILED] LbfgsbSolver with error " << rosenbrock(x) << std::endl;
        else
            std::cout << "[OK] LbfgsbSolver " << std::endl;
    }




}




#endif /* UNITTEST_HPP_ */
