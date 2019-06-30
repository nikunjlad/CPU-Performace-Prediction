%% Polynomial Regression with Regularization

%% Clearing and closing the figures
close all;
clc;
clear all;

%% Loading data set

fprintf('Loading dataset...\n\n');
load('comp.mat');
y = comp(:,8);   % assigning target column to variable

%% Feature Mapping for Polynomial Regression 

X(:,1) = comp(:,2);   % secon feature is almost linear
X(:,2) = (comp(:,3).^2) / 4;  % third feature is somewhat parabolic
X(:,3) = comp(:,7); % seventh feature is linear
X(:,4) = -1 * log(comp(:,1)) / 0.4342; % first feature is exponential

p = 3; % Polynomial degree to which columns to be raised

% Feature mapping done by poly to higher orders and sinusoids
C = poly(comp(:,[4,5,6]),p); 
X = [X C]; % creating a unified feature space
[X, ~, ~] = normalize(X); % normalize the feature data
X = [ones(size(X, 1), 1), X]; % add bias for better learning

%% Creating training and testing datasets for the learning algorithm

% Training data
Xtrain = X(1:160,:); % disp(size(Xtrain));
ytrain = y(1:160,:); % disp(size(ytrain));

% Cross validation data
Xval = X(161:190,:); % disp(size(xval));
yval = y(161:190,:); % disp(size(yval));

% Testing Data
Xtest = X(191:end,:); % disp(size(Xtest));
ytest = y(191:end,:); % disp(size(ytest));
m = length(ytrain);

%% Learning Curves (lambda = 0)

% Learning curves help us tackle bias-variance problem. If the training and
% cross validation error converge to an acceptable error level then its
% alright. The following learning curve will help us find the training and
% cross validation performance with different number of training examples

lambda = 0;
[error_train, error_val] = ...
    learningCurve(Xtrain, ytrain, ...
                  Xval, yval, ...
                  lambda);

figure;
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression (lambda = 0)')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')

%% Learning Curves (lambda = 1)

% Learning curves help us tackle bias-variance problem. If the training and
% cross validation error converge to an acceptable error level then its
% alright. The following learning curve will help us find the training and
% cross validation performance with different number of training examples

lambda = 1;
[error_train, error_val] = ...
    learningCurve(Xtrain, ytrain, ...
                  Xval, yval, ...
                  lambda);

figure;
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression (lambda = 1)')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')

%% Learning Curves (lambda = 2)

% Learning curves help us tackle bias-variance problem. If the training and
% cross validation error converge to an acceptable error level then its
% alright. The following learning curve will help us find the training and
% cross validation performance with different number of training examples

lambda = 2;
[error_train, error_val] = ...
    learningCurve(Xtrain, ytrain, ...
                  Xval, yval, ...
                  lambda);

figure;
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression (lambda = 2)')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')

%% Learning Curves (lambda = 3)

% Learning curves help us tackle bias-variance problem. If the training and
% cross validation error converge to an acceptable error level then its
% alright. The following learning curve will help us find the training and
% cross validation performance with different number of training examples

lambda = 3;
[error_train, error_val] = ...
    learningCurve(Xtrain, ytrain, ...
                  Xval, yval, ...
                  lambda);

figure;
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression (lambda = 3)')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')

%% Validation Curve 

% Validation curve help us find the optimum lambda parameter which can be
% useful for appreciable regularization. Here we learning curve obtained is
% the training and the testing curve with varying lambda values.

[lambda_vec, error_train, error_val] = ...
    validationCurve(Xtrain, ytrain, Xval, yval);

close all;
figure;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');


%%  Training Linear Regression with Regularization (lambda = 0)

lambda = 0;
[theta] = trainLinearReg(Xtrain, ytrain, lambda);

%% Training accuracy of our algorithm using linear regression

fprintf('lambda = 0\n');
% Applying learnt parameters on test data
pricetr = Xtrain * theta;

% Showing algorithm accuracy
errortr = abs(pricetr - ytrain) ./ ytrain;
accuracytr = 100 - (mean(errortr) * 100);
fprintf('Training accuracy on training set: %f\n', accuracytr);

%%  Cross Validation Accuracy of our algorithm

% disp(size(Xtest));
pricecv = Xval * theta;

% Showing algorithm accuracy
errorcv = abs(pricecv - yval) ./ yval;
accuracycv = 100 - (mean(errorcv) * 100);
fprintf('Cross Validation accuracy on cross validation set: %f\n', accuracycv);

%% Testing our learnt algorithm using linear regression

% Applying learnt parameters on test data
%disp(size(Xtest));
pricete = Xtest * theta;

% Showing algorithm accuracy
errorte = abs(pricete - ytest) ./ ytest;
accuracyte = 100 - (mean(errorte) * 100);
fprintf('Testing accuracy on test set: %f\n\n', accuracyte);


%%  Training Linear Regression with Regularization (lambda = 1)

lambda = 1;
[theta1] = trainLinearReg(Xtrain, ytrain, lambda);

%% Training accuracy of our algorithm using linear regression

fprintf('lambda = 1\n');
% Applying learnt parameters on test data
pricetr = Xtrain * theta1;

% Showing algorithm accuracy
errortr = abs(pricetr - ytrain) ./ ytrain;
accuracytr = 100 - (mean(errortr) * 100);
fprintf('Training accuracy on training set: %f\n', accuracytr);

%%  Cross Validation Accuracy of our algorithm

%disp(size(Xtest));
pricecv = Xval * theta1;

% Showing algorithm accuracy
errorcv = abs(pricecv - yval) ./ yval;
accuracycv = 100 - (mean(errorcv) * 100);
fprintf('Cross Validation accuracy on cross validation set: %f\n', accuracycv);

%% Testing our learnt algorithm using linear regression

% Applying learnt parameters on test data
%disp(size(Xtest));
pricete = Xtest * theta1;

% Showing algorithm accuracy
errorte = abs(pricete - ytest) ./ ytest;
accuracyte = 100 - (mean(errorte) * 100);
fprintf('Testing accuracy on test set: %f\n\n', accuracyte);

%%  Training Linear Regression with Regularization (lambda = 2)

lambda = 2;
[theta2] = trainLinearReg(Xtrain, ytrain, lambda);

%% Training accuracy of our algorithm using linear regression
fprintf('lambda = 2\n');
% Applying learnt parameters on test data
pricetr = Xtrain * theta2;

% Showing algorithm accuracy
errortr = abs(pricetr - ytrain) ./ ytrain;
accuracytr = 100 - (mean(errortr) * 100);
fprintf('Training accuracy on training set: %f\n', accuracytr);

%%  Cross Validation Accuracy of our algorithm

%disp(size(Xtest));
pricecv = Xval * theta2;

% Showing algorithm accuracy
errorcv = abs(pricecv - yval) ./ yval;
accuracycv = 100 - (mean(errorcv) * 100);
fprintf('Cross Validation accuracy on cross validation set: %f\n', accuracycv);

%% Testing our learnt algorithm using linear regression

% Applying learnt parameters on test data
%disp(size(Xtest));
pricete = Xtest * theta2;

% Showing algorithm accuracy
errorte = abs(pricete - ytest) ./ ytest;
accuracyte = 100 - (mean(errorte) * 100);
fprintf('Testing accuracy on test set: %f\n\n', accuracyte);

%%  Training Linear Regression with Regularization (lambda = 2.5)

lambda = 2.5;
[theta3] = trainLinearReg(Xtrain, ytrain, lambda);

%% Training accuracy of our algorithm using linear regression

fprintf('lambda = 2.5\n');
% Applying learnt parameters on test data
pricetr = Xtrain * theta3;

% Showing algorithm accuracy
errortr = abs(pricetr - ytrain) ./ ytrain;
accuracytr = 100 - (mean(errortr) * 100);
fprintf('Training accuracy on training set: %f\n', accuracytr);

%%  Cross Validation Accuracy of our algorithm

%disp(size(Xtest));
pricecv = Xval * theta3;

% Showing algorithm accuracy
errorcv = abs(pricecv - yval) ./ yval;
accuracycv = 100 - (mean(errorcv) * 100);
fprintf('Cross Validation accuracy on cross validation set: %f\n', accuracycv);

%% Testing our learnt algorithm using linear regression

% Applying learnt parameters on test data
%disp(size(Xtest));
pricete = Xtest * theta3;

% Showing algorithm accuracy
errorte = abs(pricete - ytest) ./ ytest;
accuracyte = 100 - (mean(errorte) * 100);
fprintf('Testing accuracy on test set: %f\n\n', accuracyte);