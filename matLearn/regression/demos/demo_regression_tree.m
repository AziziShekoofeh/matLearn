%% Description of demo_regression_tree.m
% Demonstrates stump and tree regression with either constant (mean) or
% linear (minimizing sum of squares) prediction for each subset of the
% training data space

clear all
close all
generateData_consGroups

%% usage of constant regression stump (consGroups data)
options_st1 = [];
options_st1.modelType = 'cns';
model_st1 = ml_regression_stump(Xtrain, ytrain, options_st1);
yhat_st1 = model_st1.predict(model_st1, Xtest);
testError_st1 = mean(abs(yhat_st1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_st1.name, testError_st1);

%% usage of constant regression tree (consGroups data)
options_tr1 = [];
options_tr1.modelType = 'cns';
options_tr1.splitSample = 'bf';
model_tr1 = ml_regression_tree(Xtrain, ytrain, options_tr1);
yhat_tr1 = model_tr1.predict(model_tr1, Xtest);
testError_tr1 = mean(abs(yhat_tr1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_tr1.name, testError_tr1);

%%
plotRegression1D(Xtrain, ytrain, model_st1, model_tr1);
title('Constant');

generateData_linGroups

%% usage of linear regression stump (linGroups data)
options_st2 = [];
options_st2.modelType = 'lin';
model_st2 = ml_regression_stump(Xtrain, ytrain, options_st2);
yhat_st2 = model_st2.predict(model_st2, Xtest);
testError_st2 = mean(abs(yhat_st2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_st2.name, testError_st2);

%% usage of linear regression tree  (linGroups data)
options_tr2 = [];
options_tr2.modelType = 'lin';
options_tr2.splitSample = 'bf';
model_tr2 = ml_regression_tree(Xtrain, ytrain, options_tr2);
yhat_tr2 = model_tr2.predict(model_tr2, Xtest);
testError_tr2 = mean(abs(yhat_tr2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_tr2.name, testError_tr2);

%%
plotRegression1D(Xtrain, ytrain, model_st2, model_tr2);
title('Linear');