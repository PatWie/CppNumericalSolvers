CppNumericalSolvers (C++11 implementation with MATLAB bindings)
===================

This repository contains some solvers written in C++11 using the Eigen3 library for linear algebra. You can use this library in **C++ and Matlab**.

There are currently the following solvers:
- gradient descent solver
- Newton descent solver (c++ only)
- BFGS solver
- L-BFGS solver
- L-BFGS-B solver (c++ only)

Additional helpful functions are

- check gradient
- compute gradient by finite differences
- compute Hessian matrix by finite differences

There is a simple "unittest" inside (minimization of the Rosenbrock function). By convention these solvers minimize a function.

#Usage in C++

You only have to define c++11-functionals for calculating the function value and gradient:


	auto function_value = [] (const Vector &x) -> double {};
	auto gradient_value = [] (const Vector x, Vector &grad) -> void {};


While I discourage you from using the numerical approximation of the gradient, the definition of the gradient functional is *optional*. To use `Newton Descent` without explicitly computing the Hessian matrix you should use `computeHessian` from `Meta.h`.
To optimize (minimize) the function you can use:


	Vector x0;                  // initial guess
	LbfgsSolver lbfgs;      // choose a solver
	lbfgs.Solve(x0,function_value,gradient_value);
	// or using a numerical approximation of the gradient
	lbfgs.Solve(x0,function_value);
	// or newton
	auto hessian_value = [&](const Vector x, Matrix & hes) -> void
	{
	    hes = Matrix::Zero(x.rows(), x.rows());
	    computeHessian(function_value, x, hes);
	};
	NewtonDescentSolver newton;      // or use newton descent
	newton.Solve(x0,function_value,gradient_value,hessian_value);


I encourage you to check you gradient by 

	checkGradient(YOUR-OBJECTIVE-FUNCTION, X, A-SAMPLE-GRADIENT-VECTOR_FROM-YOUR-FUNCTIONAL);


##full sample


	int main(void) {

		// create function
		auto rosenbrock = [] (const Vector &x) -> double {
			const double t1 = (1-x[0]);
			const double t2 = (x[1]-x[0]*x[0]);
			return   t1*t1 + 100*t2*t2;
		};

		// create derivative of function
		auto Drosenbrock = [] (const Vector x, Vector &grad) -> void {
			grad[0]  = -2*(1-x[0])+200*(x[1]-x[0]*x[0])*(-2*x[0]);
			grad[1]  = 200*(x[1]-x[0]*x[0]);
		};

		// initial guess
		Vector x0(2);x0 << 1,2;
		// get derivative
		Vector dx0(2);
		Drosenbrock(x0,dx0);
		// verify gradient by finite differences
		checkGradient(rosenbrock,x0,dx0);

		// use solver (GradientDescentSolver,BfgsSolver,LbfgsSolver,LbfgsbSolver)
		LbfgsSolver g;
		g.Solve(x0,rosenbrock,Drosenbrock);

		std::cout << std::endl<<std::endl<< x0.transpose();

		return 0;
	}


# Usage in MATLAB (beta)
This is just a beta. If something goes wrong MATLAB will tell it to you by crashing.

- Checkout the `matlab-bindings` branch using git

	`cd PATH_TO_THIS_LIBRARY` and 
	`git pull origin matlab` and
	`git checkout matlab`


- Download Eigen and copy the `Eigen` folder into `matlab-bindings`. The compilation script will check if Eigen exists.
- Mex files are no fun! But if your configs are correct (reminder: set -std=c++11) you can run `make` inside the `matlab-bindings` folder. This should create a file name `cppsolver.mexw64` or `cppsolver.mexw32`.

To solve your problem just define the objective as a function like

	function y = rosenbrock( x )
	t1 = (1-x(1));
	t2 = (x(2)-x(1)*x(1));
	y =t1*t1 + 100*t2*t2; 
	end

an save it under `rosenbrock.m` and run

	solution=cppsolver([-1;2],@rosenbrock)
	% or if you specify a gradient:
	cppsolver([-1;2],@rosenbrock,'gradient',@rosenbrock_grad)



If you pass a gradient function the mex-file will check the gradient function in the initialization point to make sure you have no typos in your gradient function. You can skip this check by calling 

	solution=cppsolver([-1;2],@rosenbrock,'gradient',@rosenbrock_grad,'skip_gradient_check','true')


The default solver is `L-BFGS`. To specify another solver you can call

	solution=cppsolver([-1;2],@rosenbrock,'gradient',@rosenbrock_grad,'solver','THE-SOLVER-TYPE')

where you can select a solver from:

- l-bfgs
- bfgs
- gradientdescent

The other solver have not matlab bindings yet. 
There is an example in the folder `matlab-bindings`. The output should be

	cppsolver([-1;2],@rosenbrock,'gradient',@rosenbrock_grad);
	------------------------------------------------------------
	Found objective function: rosenbrock
	parsing key: gradient
	Found gradient function: rosenbrock_grad
	Elapsed time is 0.106917 seconds.
	cppsolver([-1;2],@rosenbrock);
	------------------------------------------------------------
	Found objective function: rosenbrock
	Elapsed time is 0.014738 seconds.
	fminsearch(@rosenbrock,[-1;2]);
	------------------------------------------------------------
	Elapsed time is 0.094601 seconds.

Notice that specifying a gradient increase the optimization speed. Notice, that fminsearch will use caches if you run this file multiple times.

#License

	Copyright (c) 2014 Patrick Wieschollek
	url: https://github.com/PatWie/CppNumericalSolvers

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.


Rebase - Warning
-----------------
The commit `3122fa1710a0f592b99938ad728c59d787c3077a` was a rebase of the old history to clean things up. Fork this project again, if you have trouble to pull commits. I will never do a rebase on this repository again. I promise.