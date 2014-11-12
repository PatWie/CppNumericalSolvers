% we minimize the rosenbrock function

fprintf('\n\n L-BFGS starting in [-1,2]\n');
tic
solution = cppsolver([-1;2],@rosenbrock,'gradient',@rosenbrock_grad)
toc

fprintf('\n------------------------------------------------------------\n\n');
fprintf('\n\n L-BFGS starting in [2,2] (without gradient, this is a WARNING!)\n');
tic
solution = cppsolver([2;2],@rosenbrock)
toc

fprintf('\n------------------------------------------------------------\n\n');
fprintf('\n\n NEWTON with given hessian and gradient starting in [-1,2]\n');
tic
solution = cppsolver([-1;2],@rosenbrock,'gradient',@rosenbrock_grad,'hessian',@rosenbrock_hessian,'solver','newton')
toc

fprintf('\n------------------------------------------------------------\n\n');
fprintf('fminsearch starting in [-1,2]\n');
tic
solution = fminsearch(@rosenbrock,[-1;2])
toc

fprintf('\n------------------------------------------------------------\n\n');
fprintf('\n\fminunc starting in [-1,2] \n');
tic
solution = fminunc(@rosenbrock,[-1;2])
toc

fprintf('\n------------------------------------------------------------\n\n');
fprintf('\n\n BFGS starting in [-1,2] \n');
tic
solution = cppsolver([-1;2],@rosenbrock,'gradient',@rosenbrock_grad,'solver','bfgs')
toc