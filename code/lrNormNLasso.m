%% Linear Regression with Normal Equations Algorithm (without Lasso)

%% Clearing and closing the figures
close all;
clc;
clear all;

%% Loading data set and visualizing it

fprintf('Loading dataset...\n\n');
load('comp.mat');
X = comp(:,1:7);
y = comp(:,8);

[X, mu, sigma] = normalize(X);
% adding intercept term
X = [ones(size(X,1),1) X];

%% Creating training and testing datasets for the learning algorithm

Xtrain = X(1:180,:); % disp(size(Xtrain));
ytrain = comp(1:180,8); % disp(size(ytrain));

Xtest = X(181:end,:); % disp(size(Xtest));
ytest = comp(181:end,8); % disp(size(ytest));

%% Applying normal equations

fprintf('Linear Regression using Normal equations....\n');
% Acquiring parameters using normal equations
[theta] = normalEqn(Xtrain, ytrain);

%% Training accuracy of our algorithm using Normal equations

% Applying learnt parameters on test data
pricetr = Xtrain * theta;

% Showing algorithm accuracy
errortr = abs(pricetr - ytrain) ./ ytrain; % error between actual and predicted
accuracytr = 100 - (mean(errortr) * 100); % percentage accuracy obtained
fprintf('Training accuracy on training set: %f\n', accuracytr);

%% Testing our learnt algorithm using Normal equations

% Applying learnt parameters on test data
pricete = Xtest * theta;

% Showing algorithm accuracy
errorte = abs(pricete - ytest) ./ ytest;  % error between actual and predicted values
accuracyte = 100 - (mean(errorte) * 100); % percentage accuracy obtained
fprintf('Testing accuracy on test set: %f\n', accuracyte);