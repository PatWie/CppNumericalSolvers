function [f,g,h] = rosenbrock_struct(x)
% Calculate objective f
f = 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;

if nargout > 1 % gradient required
    g = [-400*(x(2)-x(1)^2)*x(1)-2*(1-x(1));
        200*(x(2)-x(1)^2)];
if nargout > 2 % hessian required
    h = [ 1200*x(1)*x(1)-400*x(2)+1 , -400*x(1); -400*x(1) , 200];
end