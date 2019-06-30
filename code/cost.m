function J = cost(X, y, theta)
%   Compute cost for linear regression with multiple variables
%   J = cost(X, y, theta) computes the cost using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

%% Cost computation
predictions = X * theta;
squaredError = (predictions - y) .^ 2;
J = 1 / (2 * m) * sum(squaredError);

end