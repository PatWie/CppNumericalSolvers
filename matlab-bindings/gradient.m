function grad = gradient( x )
grad = [-2*(1-x(0))+200*(x(1)-x(0)*x(0))*(-2*x(0))    ;
         200*(x(1)-x(0)*x(0))];
end

