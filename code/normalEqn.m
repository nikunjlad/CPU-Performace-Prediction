function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

%% Normal equations is an advanced optimization step to calculate the parameter values

theta = pinv(X' * X) * X' * y;

end