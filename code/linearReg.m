%% Linear Regression with Regularization

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

lambda = 0;
[error_train, error_val] = ...
    learningCurve(Xtrain, ytrain, ...
                  Xval, yval, ...
                  lambda);

              figure;
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
fprintf('Program paused. Press enter to continue.\n');
pause;

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
title('Learning curve for linear regression (lambda = 1)');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');

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
title('Learning curve for linear regression (lambda = 2)');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');

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
title('Learning curve for linear regression (lambda = 3)');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');

%% Validation Curve 

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

% Applying learnt parameters on test data
pricetr = Xtrain * theta;

% Showing algorithm accuracy
errortr = abs(pricetr - ytrain) ./ ytrain;
accuracytr = 100 - (mean(errortr) * 100);
fprintf('Training accuracy on training set: %f\n', accuracytr);

%% Cross Validation Accuracy of our algorithm

% Applying learnt parameters on test data
pricecv = Xval * theta;

% Showing algorithm accuracy
errorcv = abs(pricecv - yval) ./ yval;
accuracycv = 100 - (mean(errorcv) * 100);
fprintf('Cross Validation accuracy on cross validation set: %f\n', accuracycv);

%% Testing our learnt algorithm using linear regression

% Applying learnt parameters on test data
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

pricecv = Xval * theta2;

% Showing algorithm accuracy
errorcv = abs(pricecv - yval) ./ yval;
accuracycv = 100 - (mean(errorcv) * 100);
fprintf('Cross Validation accuracy on cross validation set: %f\n', accuracycv);

%% Testing our learnt algorithm using linear regression

% Applying learnt parameters on test data
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
pricete = Xtest * theta3;

% Showing algorithm accuracy
errorte = abs(pricete - ytest) ./ ytest;
accuracyte = 100 - (mean(errorte) * 100);
fprintf('Testing accuracy on test set: %f\n\n', accuracyte);



