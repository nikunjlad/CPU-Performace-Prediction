function [J, grad] = regCost(X, y, theta, lambda)
%   Compute cost and gradient for regularized linear 
%   regression with multiple variables
%   [J, grad] = regCost(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

%% Calculating the cost and the gradients

% The following is same as calculating cost and gradients, as in cost.m,
% the only difference being that the below computations are matrix
% operations rather than loop operations. Hence we need not iterate over
% columns and compute cost by independently summing and adding elements.
% The inherent nature of matrix operations handles the same and hence the
% code becomes more robust.

h = X * theta; % calculating the hypothesis value
J = (1 / (2 * m)) * (sum((h - y) .^ 2) + lambda * sum(theta(2:end,:) .^ 2)); % Calculate the cost
grad = (1 / m) * ((X' * (h - y)) + lambda * [zeros(size(theta,2),1); theta(2:end,:)]); % Calculate the gradients

grad = grad(:);

end