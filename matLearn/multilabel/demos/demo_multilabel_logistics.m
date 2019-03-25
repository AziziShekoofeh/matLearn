%% Description of demo_multilabel_logistics.m
% Demonstrates use of independent logistic regression classifiers for each 
% candidate class for multilabel classification. Each combination labels is
% represented by a unique color in the output plot and as a unique integer
% in [1,2^N] where N is the number of classes
clear all
close all
generateData_multiLabel

%% usage of independent logistic regression
options = struct('nLabels',nLabels);
model = ml_multilabel_independent(Xtrain,ytrain,options);
yhatTest = model.predict(model, Xtest);
yhatTrain = model.predict(model, Xtrain);
testError = sum(ytest~=yhatTest)/length(ytest);
model.trainError = sum(ytrain~=yhatTrain)/length(ytrain);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model.name, testError);
linear_makeOneContourPlot(Xtrain,ytrain, model)

%% usage of independent logistic regression with L2-regularization
options = struct('nLabels',nLabels,'lambdaL2',1e-4);
model = ml_multilabel_independent(Xtrain,ytrain,options);
yhatTest = model.predict(model, Xtest);
yhatTrain = model.predict(model, Xtrain);
testError = sum(ytest~=yhatTest)/length(ytest);
model.trainError = sum(ytrain~=yhatTrain)/length(ytrain);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model.name, testError);
linear_makeOneContourPlot(Xtrain,ytrain, model)

