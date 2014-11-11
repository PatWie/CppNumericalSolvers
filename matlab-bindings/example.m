% we minimize the rosenbrock function
fprintf('\n\ncppsolver([-1;2],@rosenbrock,''gradient'',@rosenbrock_grad);\n');
fprintf('------------------------------------------------------------\n\n');
tic
solution = cppsolver([-1;2],@rosenbrock,'gradient',@rosenbrock_grad);
toc

fprintf('\n\ncppsolver([-1;2],@rosenbrock);\n');
fprintf('------------------------------------------------------------\n\n');
tic
solution = cppsolver([-1;2],@rosenbrock);
toc

fprintf('\n\nfminsearch(@rosenbrock,[-1;2]);\n');
fprintf('------------------------------------------------------------\n\n');
tic
solution = fminsearch(@rosenbrock,[-1;2]);
toc