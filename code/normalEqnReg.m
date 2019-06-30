function [theta] = normalEqnReg(X, y, lambda)
%   Computes the closed-form solution to linear regression 
%   NORMALEQNREG(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

%% Normal equations is an advanced optimization step to calculate the parameter values

a = eye(size(X,2));
a(1,1) = 0;
theta = pinv((X' * X) + lambda * a) * X' * y;

end