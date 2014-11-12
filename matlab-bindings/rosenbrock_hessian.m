function hess = rosenbrock_hessian( x )

hess = [ 1200*x(1)*x(1)-400*x(2)+1 , -400*x(1); -400*x(1) , 200];

end

