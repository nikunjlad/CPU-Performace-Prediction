function [X_norm, mu, sigma] = normalize(X)
%   Normalizes the features in X 
%   normalize(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly

%% Normalizing the features using mean normalization
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
% disp(size(mu)); disp(size(sigma)); disp(size(X_norm));

for i = 1:size(X,2)
    mu = mean(X(:,i));
    sigma = std(X(:,i));
    X_norm(:,i) = (X(:,i) - mu) ./ sigma;
end
end