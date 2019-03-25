%% Description of demo_multiclass_binarysubclassifiers.m
% Multiclass classification using binary classification subclassifiers with
% 1-vs-1 or 1-vs-all design

clear all
close all
generateData_5grid

%% usage of 1-vs-all algorithm on binary logistic classifier
options_1va = [];
options_1va.subModel = @ml_binaryclass_logistic;
model_1va = ml_multiclass_1vA(Xtrain, ytrain, options_1va);
yhat_1va = model_1va.predict(model_1va, Xtest);
testError_1va = mean(yhat_1va ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_1va.name, testError_1va);

%% usage of 1-vs-1 algorithm on binary logistic classifier
options_1v1 = [];
options_1v1.subModel = @ml_binaryclass_logistic;
model_1v1 = ml_multiclass_1v1(Xtrain, ytrain, options_1v1);
yhat_1v1 = model_1v1.predict(model_1v1, Xtest);
testError_1v1 = mean(yhat_1va ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_1v1.name, testError_1v1);

%%
figure;
plotClassifier(Xtrain, ytrain, model_1va);
figure;
plotClassifier(Xtrain, ytrain, model_1v1);
