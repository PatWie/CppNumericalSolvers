
x0 =[-1 2]';
warning('off','optim:fminunc:SwitchingMethod')
fprintf('x0              solver            f(x*)     x*              time    \n');
fprintf('--------------------------------------------------------------------\n');

%% --------------------------------------------------------------------------------
fprintf('\n(finite gradient)\n')
solver = {'gradientdescent','cg','bfgs','l-bfgs','newton'};
for s=1:numel(solver)
  tic
  [fx,x] = cppoptlib(x0,@rosenbrock,'solver',solver{s});
  t = toc;
  fprintf('(%02.2f,%02.2f)    %-15s   %03.4f   (%06.4f,%06.4f)  %f \n', ...
    x0(1),x0(2), solver{s}, fx,x(1),x(2),t);
end
tic
x = fminsearch(@rosenbrockstruc,x0);
t = toc;
fprintf('(%02.2f,%02.2f)    %-15s   %03.4f   (%06.4f,%06.4f)  %f \n', ...
    x0(1),x0(2), 'fminsearch', rosenbrock(x),x(1),x(2),t);
tic
x = fminunc(@rosenbrockstruc,x0);
t = toc;
fprintf('(%02.2f,%02.2f)    %-15s   %03.4f   (%06.4f,%06.4f)  %f \n', ...
    x0(1),x0(2), 'fminunc', rosenbrock(x),x(1),x(2),t);

%% --------------------------------------------------------------------------------
fprintf('\n(with gradient)\n')
solver = {'gradientdescent','cg','bfgs','l-bfgs','l-bfgs-b','newton'};
for s=1:numel(solver)
  tic
  [fx,x] = cppoptlib(x0,@rosenbrock,'gradient',@rosenbrock_grad,'solver',solver{s});
  t = toc;
  fprintf('(%02.2f,%02.2f)    %-15s   %03.4f   (%06.4f,%06.4f)  %f \n', ...
    x0(1),x0(2), solver{s}, fx,x(1),x(2),t);
end

%% --------------------------------------------------------------------------------
fprintf('\n(finite hessian)\n')
solver = {'newton'};
for s=1:numel(solver)
  tic
  [fx,x] = cppoptlib(x0,@rosenbrock,'gradient',@rosenbrock_grad,'solver',solver{s});
  t = toc;
  fprintf('(%02.2f,%02.2f)    %-15s   %03.4f   (%06.4f,%06.4f)  %f \n', ...
    x0(1),x0(2), solver{s}, fx,x(1),x(2),t);
end

%% --------------------------------------------------------------------------------
fprintf('\n(with hessian)\n')
solver = {'newton'};
for s=1:numel(solver)
  tic
  [fx,x] = cppoptlib(x0,@rosenbrock,'gradient',@rosenbrock_grad,'hessian',@rosenbrock_hessian,'solver',solver{s});
  t = toc;
  fprintf('(%02.2f,%02.2f)    %-15s   %03.4f   (%06.4f,%06.4f)  %f \n', ...
    x0(1),x0(2), solver{s}, fx,x(1),x(2),t);
end

warning('on','optim:fminunc:SwitchingMethod')