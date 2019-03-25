%% Description demo_regression_regressOnOne.m
% Demonstration of regression on a single explanatory variable with
% regression to the mean as a baseline.

clear all
close all

% Load synthetic {Xtrain,ytrain} and {Xtest,ytest}
load data_regressOnOne.mat

%% Mean model

% Train mean model
options = [];
[model_mean] = ml_regression_mean(Xtrain,ytrain,options);

% Test mean model
yhat = model_mean.predict(model_mean,Xtest);

% Measure test error
testError = mean(abs(yhat-ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n',model_mean.name,testError);

%% Regress on One model
options = [];
options.selectedFeature = 1;
[model_regressOnOne] = ml_regression_regressOnOne(Xtrain,ytrain,options);

yhat = model_regressOnOne.predict(model_regressOnOne,Xtest);

% Measure test error
testError = mean(abs(yhat-ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n',model_regressOnOne.name,testError);


%% Plot the performance of both models
plotRegression1D(Xtrain,ytrain,model_mean,model_regressOnOne);
