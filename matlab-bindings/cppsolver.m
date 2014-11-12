function [x, fx] = cppsolver(x0, @objective, [args])
% CPPSOLVER  minimize a given objective, when starting in x0
%   x = CPPSOLVER(x0, @objective, [args]) returns the argmin
%   [x, fx]  = CPPSOLVER(x0, @objective, [args]) returns the argmin and min
%
%   https://github.com/PatWie/CppNumericalSolvers
%
%
%   ARGUMENTS:
%
%   "x0"                   :  intial guess
%   "@objective"           :  handle of Matlab-function that returns the objective value in x
% 
%   The following arguments are optional
%
%   "solver"               :  selects the solving algorithm
%       -   "gradientdescent"    , gradient descent
%       -   "newton"             , newton descent
%       *   "bfgs"               , Broyden-Fletcher–Goldfarb-Shanno
%       -   "l-bfgs"             , limited Broyden-Fletcher–Goldfarb-Shanno 
%       -   "l-bfgs-b"           , limited Broyden-Fletcher–Goldfarb-Shanno with bounds
% 
%       example "(...,'solver','l-bfgs')"
%
%   "@gradient"            :  handle of Matlab-function that returns the gradient in x
%   "@hessian"             :  handle of Matlab-function that returns the hessian  in x
%   "skip_gradient_check"  :  specify whether the gradient should be verfied once
%       *   "true"               , check || numerical approximation - given gradient ||_2    < 1e-5
%       -   "false"              , skip check
%
%       example "(...,'skip_gradient_check','true')"
%
%   "skip_hessian_check"   :  specify whether the hessian should be verfied once
%       -   "true"               , check || numerical approximation - given hessian  ||_frob < 1e-3
%       *   "false"              , skip check
%
%       example "(...,'skip_hessian_check','false')"
%
%   "ub"                   :  define an upper bound (for L-BFGS-b)
%
%       example "(...,'ub',[3;3;3])"
%
%   "lb"                   :  define a lower bound (for L-BFGS-b)
%
%       example "(...,'lb',[-1;-1;-1])"
%
%
%  EXAMPLES:
%   Assume that the objective function f: IR^2 -> IR is a Matlab-function called "objective"
%          that the gradient  function f: IR^2 -> IR^2 is a Matlab-function called "gradient"
%          that the hessian   function f: IR^2 -> IR^(2x2) is a Matlab-function called "hessian"
% 
%   simple call (bfgs)
%      x0      = [0;0];
%      [x, fx] = cppsolver(x0,@objective)
%
%   Newton-Descent
%      x0      = [0;0];
%      [x, fx] = cppsolver(x0,@objective,'gradient',@gradient,'hessian',@hessian)
%
%   L-BFGS-B
%      x0      = [0;0];
%      ub      = [100;150];
%      lb      = [0;1];
%      [x, fx] = cppsolver(x0,@objective,'gradient',@gradient,'ub',ub,'lb',lb)