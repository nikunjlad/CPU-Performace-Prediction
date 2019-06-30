function [theta, J] = gradients(X, y, theta, alpha, iterations)
%   Performs gradient descent to learn theta
%   theta = gradients(x, y, theta, alpha, iterations) updates theta by
%   taking iterations gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J = zeros(iterations, 1);

%% Performing gradient descent to achive global optimum
for iter = 1:iterations
    
    predictions = X * theta;
    for j=1:1:length(theta)
		theta(j)=theta(j) - alpha*sum(((predictions - y) .* X(:,j))) / m;
    end

    % Save the cost J in every iteration    
    J(iter) = cost(X, y, theta);

end

end