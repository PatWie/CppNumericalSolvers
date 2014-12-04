/*
 * main.cpp
 *      Author: Patrick Wieschollek
 */
#include <iostream>
#include <functional>
#include "LbfgsSolver.h"

int example000(void);
int example001(void);
int example002(void);
int example003(void);

using namespace pwie;

int main(void)
{
    example000();
    example001();
    example002();
    example003();

    return 0;
}



int example000(void)
{

    std::cout << std::endl << std::endl
              << "example checkgradient:    " << std::endl
              << "------------------------------------------" << std::endl;
    // check your gradient!
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
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1]  = 200 * (x[1] - x[0] * x[0]);
    };

    // create derivative of function
    auto WrongDrosenbrock = [](const Vector x, Vector & grad) -> void
    {
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (2 * x[0]); // <<-- last term wrong sign
        grad[1]  = 200 * (x[1] - x[0] * x[0]);
    };

    // random x
    Vector x(2);
    x << 15, 8;
    // get "derivatives"
    Vector dx(2);
    Drosenbrock(x, dx);
    Vector dx_wrong(2);
    WrongDrosenbrock(x, dx_wrong);
    // verify gradient by finite differences
    std::cout << "check correct gradient: ";
    if(checkGradient(rosenbrock, x, dx))
    {
        std::cout << "gradient probably correct" << std::endl;
    }
    std::cout << "check wrong gradient:   ";
    checkGradient(rosenbrock, x, dx_wrong);


    return 0;
}

int example001(void)
{

    std::cout << std::endl << std::endl
              << "example optimization without gradient:    " << std::endl
              << "------------------------------------------" << std::endl;

    // minimize rosenbrock function only using the objective function
    auto objectiveFunction = [](const Vector & x) -> double
    {
        // rosenbrock
        const double t1 = (1 - x[0]);
        const double t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    };


    // initial guess
    Vector x0(2);
    x0 << 3, 2;
    // use solver (GradientDescentSolver,BfgsSolver,LbfgsSolver,LbfgsbSolver)
    LbfgsSolver g;
    g.solve(x0, objectiveFunction);

    std::cout << std::endl << std::endl;
    std::cout << "result:    " << x0.transpose();
    std::cout << std::endl;
    std::cout << "should be: " << "1 1" << std::endl;


    return 0;
}


int example002(void)
{

    std::cout << std::endl << std::endl
              << "example optimization with  gradient:    " << std::endl
              << "------------------------------------------" << std::endl;

    // minimize rosenbrock function using predefined objective function and vector of partial derivatives
    auto objectiveFunction = [](const Vector & x) -> double
    {
        // rosenbrock
        const double t1 = (1 - x[0]);
        const double t2 = (x[1] - x[0] * x[0]);
        return   t1 * t1 + 100 * t2 * t2;
    };

    // create derivative of function
    auto partialDerivatives = [](const Vector x, Vector & grad) -> void
    {
        grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
        grad[1]  = 200 * (x[1] - x[0] * x[0]);
    };


    // initial guess
    Vector x0(2);
    x0 << 15, 8;
    // use solver (GradientDescentSolver,BfgsSolver,LbfgsSolver,LbfgsbSolver)
    LbfgsSolver g;
    g.solve(x0, objectiveFunction, partialDerivatives);

    std::cout << std::endl << std::endl;
    std::cout << "result:    " << x0.transpose();
    std::cout << std::endl;
    std::cout << "should be: " << "1 1" << std::endl;



    return 0;
}


int example003(void)
{

    std::cout << std::endl << std::endl
              << "example optimization least squares:    " << std::endl
              << "------------------------------------------" << std::endl;

    Matrix A(3, 2);
    A << 0.2, 0.25, 0.4, 0.5, 0.4, 0.25;
    Vector y(3);
    y << 0.9, 1.7, 1.2;

    // least squares
    auto objectiveFunction = [&](const Vector & x) -> double
    {
        Vector tmp = A * x - y;
        return tmp.dot(tmp);
    };

    // create derivative of function
    auto partialDerivatives = [&](const Vector x, Vector & grad) -> void
    {
        grad = Vector(x.rows());
        grad = 2 * A.transpose() * (A * x) - 2 * A.transpose() * y;
    };


    // initial guess
    Vector x0(2);
    x0 << 15, 8;
    // use solver (GradientDescentSolver,BfgsSolver,LbfgsSolver,LbfgsbSolver)
    LbfgsSolver g;
    g.solve(x0, objectiveFunction, partialDerivatives);

    std::cout << std::endl << std::endl;
    std::cout << "result:    " << x0.transpose();
    std::cout << std::endl;
    std::cout << "should be:  " << "1.7 2.08" << std::endl;



    return 0;
}
