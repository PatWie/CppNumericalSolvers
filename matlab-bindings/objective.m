function y = objective( x )
t1 = (1-x(0));
t2 = (x(1)-x(0)*x(0));
y =t1*t1 + 100*t2*t2; 
end

