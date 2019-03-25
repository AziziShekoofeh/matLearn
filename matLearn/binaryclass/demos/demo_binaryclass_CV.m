%% Description of demo_binaryclass_CV.m
% Demonstrates usage of ml_general_CV on model parameters for binary
% classification, using logistic regression under RBF basis as exemplar

clear all
close all
generateData_vert

%% usage of RBF basis logistic regression
options_bs = [];
options_bs.subModel = @ml_binaryclass_logistic;
options_bs.subOptions.addBias = 1;
options_bs.subOptions.lambdaL2 = 7;
options_bs.basisFunc = @ml_kernel_rbf;
options_bs.basisOptions.sigma = 2;
model_bs = ml_binaryclass_basis(Xtrain, ytrain, options_bs);
yhat_bs = model_bs.predict(model_bs, Xtest);
testError_bs = mean(yhat_bs ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_bs.name, testError_bs);
figure;
plot2DClassifier(Xtrain, ytrain, model_bs)

%% usage of RBF basis logistic regression with CV on sigma
options_cv1 = [];
options_cv1.subModel = @ml_binaryclass_basis;
options_cv1.subOptions.subModel = @ml_binaryclass_logistic;
options_cv1.subOptions.subOptions.addBias = 1;
options_cv1.subOptions.subOptions.lambdaL2 = 2;
options_cv1.subOptions.basisFunc = @ml_kernel_rbf;
options_cv1.paramNames = 'basisOptions.sigma';
options_cv1.paramValues = [0.5 1 1.5 2 2.5]';
options_cv1.nParams = 1;
options_cv1.loss = 'mc';
model_cv1 = ml_general_CV(Xtrain, ytrain, options_cv1);
yhat_cv1 = model_cv1.predict(model_cv1, Xtest);
testError_cv1 = mean(yhat_cv1 ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_cv1.name, testError_cv1);
fprintf('Best %s is: %.3f\n', options_cv1.paramNames, model_cv1.bestParams);
figure;
plot2DClassifier(Xtrain, ytrain, model_cv1)

%% usage of sigmoid basis logistic regression with CV on sigma and lambda (L2 regularizer)
options_cv2 = [];
options_cv2.subModel = @ml_binaryclass_basis;
options_cv2.subOptions.subModel = @ml_binaryclass_logistic;
options_cv2.subOptions.subOptions.addBias = 1;
options_cv2.subOptions.basisFunc = @ml_kernel_rbf;
options_cv2.paramNames = {'basisOptions.sigma','subOptions.lambdaL2'};
options_cv2.paramValues = {[0.5 1 1.5 2 2.5], [0.5 1 1.5 2 2.5]};
options_cv2.nParams = 2;
options_cv2.loss = 'mc';
model_cv2 = ml_general_CV(Xtrain, ytrain, options_cv2);
yhat_cv2 = model_cv2.predict(model_cv2, Xtest);
testError_cv2 = mean(yhat_cv2 ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_cv2.name, testError_cv2);
fprintf('Best %s is: %.3f\n', options_cv2.paramNames{1}, ...
        model_cv2.bestParams(1));
fprintf('Best %s is: %.3f\n', options_cv2.paramNames{2}, ...
        model_cv2.bestParams(2));
figure;
plot2DClassifier(Xtrain, ytrain, model_cv2);

