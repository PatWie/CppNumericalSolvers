function grad = rosenbrock_grad( x )
grad = [-2*(1-x(1))+200*(x(2)-x(1)*x(1))*(-2*x(1))    ;
         200*(x(2)-x(1)*x(1))];
end