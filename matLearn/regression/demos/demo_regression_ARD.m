%% Description of demo_regression_ARD.m
% Regression using Automatic Relevance Determination to encourage sparsity
% in learned weight vector with high-dimensional data where most features 
% are uninformative. L2 regression with and without regularization and
% L1 regression are presented for comparisons

clear all
close all
generateData_irrelevFeatures

%% usage of ARD regression
options_ard = [];
options_ard.addBias = 1;
options_ard.variance = 0.5;
model_ard = ml_regression_ARD(Xtrain, ytrain, options_ard);
yhat_ard = model_ard.predict(model_ard, Xtest);
testError_ard = mean(abs(yhat_ard - ytest));
fprintf('Averaged absolute test error with %s and Var %.3f is: %.3f\n', ...
        model_ard.name, options_ard.variance, testError_ard);
fprintf('Number of features used is is: %.3f\n', ...
        sum(abs(model_ard.w(2:end)) > 0.05));

%% usage of L2 regression
options_l2_1 = [];
options_l2_1.addBias = 1;
model_l2_1 = ml_regression_L2(Xtrain, ytrain, options_l2_1);
yhat_l2_1 = model_l2_1.predict(model_l2_1, Xtest);
testError_l2_1 = mean(abs(yhat_l2_1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_l2_1.name, testError_l2_1);
fprintf('Number of features used is is: %.3f\n', ...
        sum(abs(model_l2_1.w(2:end)) > 0.05));

%% usage of L2 regression with L2 regularization
options_l2_2 = [];
options_l2_2.addBias = 1;
options_l2_2.lambdaL2 = 5;
model_l2_2 = ml_regression_L2(Xtrain, ytrain, options_l2_2);
yhat_l2_2 = model_l2_2.predict(model_l2_2, Xtest);
testError_l2_2 = mean(abs(yhat_l2_2 - ytest));
fprintf('Averaged absolute test error with %s and L2 Reg is: %.3f\n', ...
        model_l2_2.name, testError_l2_2);
fprintf('Number of features used is is: %.3f\n', ...
        sum(abs(model_l2_2.w(2:end)) > 0.05));

%% usage of L1 regression
options_l1 = [];
options_l1.addBias = 1;
model_l1 = ml_regression_L1(Xtrain, ytrain, options_l1);
yhat_l1 = model_l1.predict(model_l1, Xtest);
testError_l1 = mean(abs(yhat_l1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_l1.name, testError_l1);
fprintf('Number of features used is is: %.3f\n', ...
        sum(abs(model_l1.w(2:end)) > 0.05));