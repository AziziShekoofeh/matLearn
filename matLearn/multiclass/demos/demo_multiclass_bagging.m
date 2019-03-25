%% Description of demo_multiclass_bagging.m
% Demonstrates bagging of multinomial logistic regression classifiers for a
% multiclass classification task

clear all
close all
generateData_gridMulti

%% usage of multi-class logistic regression
options_lg = [];
options_lg.addBias = 1;
model_lg = ml_multiclass_logistic(Xtrain, ytrain, options_lg);
yhat_lg = model_lg.predict(model_lg, Xtest);
testError_lg = mean(yhat_lg ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model_lg.name, testError_lg);

%% usage of multi-class logistic regression with bagging
options_bg = [];
options_bg.nModels = 20;
options_bg.subModel = @ml_multiclass_logistic;
options_bg.subOptions.addBias = 1;
model_bg = ml_multiclass_bagging(Xtrain, ytrain, options_bg);
yhat_bg = model_bg.predict(model_bg, Xtest);
testError_bg = mean(yhat_bg ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model_bg.name, testError_bg)

%%
figure;
plotClassifier(Xtrain, ytrain, model_lg);
figure;
plotClassifier(Xtrain, ytrain, model_bg);
