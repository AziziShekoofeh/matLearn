%% Description of demo_regresion_basis
% L2 regression under standard, polynomial, and RBF bases on a variety of
% datasets

clear all
close all
generateData_quad

%% usage of L2 regression (quad data)
options_l2 = [];
options_l2.addBias = 1;
model_l2 = ml_regression_L2(Xtrain, ytrain, options_l2);
yhat_l2 = model_l2.predict(model_l2, Xtest);
testError_l2 = mean(abs(yhat_l2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_l2.name, testError_l2);

%% usage of L2 regression with polynomial basis (quad data)
options_bs1 = [];
options_bs1.basisFunc = @ml_kernel_poly;
options_bs1.basisOptions.bias = 0;
options_bs1.basisOptions.order = 2;
options_bs1.subModel = @ml_regression_L2;
options_bs1.subOptions.addBias = 1;
model_bs1 = ml_regression_basis(Xtrain, ytrain, options_bs1);
yhat_bs1 = model_bs1.predict(model_bs1, Xtest);
testError_bs1 = mean(abs(yhat_bs1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_bs1.name, testError_bs1);

%% usage of L2 regression with RBF basis (quad data)
options_bs2 = [];
options_bs2.basisFunc = @ml_kernel_rbf;
options_bs2.basisOptions.sigma = 1;
options_bs2.subModel = @ml_regression_L2;
options_bs2.subOptions.addBias = 1;
model_bs2 = ml_regression_basis(Xtrain, ytrain, options_bs2);
yhat_bs2 = model_bs2.predict(model_bs2, Xtest);
testError_bs2 = mean(abs(yhat_bs2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_bs2.name, testError_bs2);

%%
plotRegression1D(Xtrain, ytrain, model_l2, model_bs1, model_bs2);
title('Quadratic');

generateData_sigmoid

%% usage of L2 regression (sigmoid data)
options_l2 = [];
options_l2.addBias = 1;
model_l2 = ml_regression_L2(Xtrain, ytrain, options_l2);
yhat_l2 = model_l2.predict(model_l2, Xtest);
testError_l2 = mean(abs(yhat_l2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_l2.name, testError_l2);

%% usage of L2 regression with polynomial basis (sigmoid data)
options_bs1 = [];
options_bs1.basisFunc = @ml_kernel_poly;
options_bs1.basisOptions.bias = 10;
options_bs1.basisOptions.order = 3;
options_bs1.subModel = @ml_regression_L2;
options_bs1.subOptions.addBias = 1;
model_bs1 = ml_regression_basis(Xtrain, ytrain, options_bs1);
yhat_bs1 = model_bs1.predict(model_bs1, Xtest);
testError_bs1 = mean(abs(yhat_bs1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_bs1.name, testError_bs1);

%% usage of L2 regression with RBF basis (sigmoid data)
options_bs2 = [];
options_bs2.basisFunc = @ml_kernel_rbf;
options_bs2.basisOptions.sigma = 1;
options_bs2.subModel = @ml_regression_L2;
options_bs2.subOptions.addBias = 1;
model_bs2 = ml_regression_basis(Xtrain, ytrain, options_bs2);
yhat_bs2 = model_bs2.predict(model_bs2, Xtest);
testError_bs2 = mean(abs(yhat_bs2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_bs2.name, testError_bs2);

%%
plotRegression1D(Xtrain, ytrain, model_l2, model_bs1, model_bs2);
title('Sigmoid');

generateData_Gauss

%% usage of L2 regression (Gauss data)
options_l2 = [];
options_l2.addBias = 1;
model_l2 = ml_regression_L2(Xtrain, ytrain, options_l2);
yhat_l2 = model_l2.predict(model_l2, Xtest);
testError_l2 = mean(abs(yhat_l2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_l2.name, testError_l2);

%% usage of L2 regression with polynomial basis (Gauss data)
options_bs1 = [];
options_bs1.basisFunc = @ml_kernel_poly;
options_bs1.basisOptions.bias = 0;
options_bs1.basisOptions.order = 2;
options_bs1.subModel = @ml_regression_L2;
options_bs1.subOptions.addBias = 1;
model_bs1 = ml_regression_basis(Xtrain, ytrain, options_bs1);
yhat_bs1 = model_bs1.predict(model_bs1, Xtest);
testError_bs1 = mean(abs(yhat_bs1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_bs1.name, testError_bs1);

%% usage of L2 regression with RBF basis (Gauss data)
options_bs2 = [];
options_bs2.basisFunc = @ml_kernel_rbf;
options_bs2.basisOptions.sigma = 1;
options_bs2.subModel = @ml_regression_L2;
options_bs2.subOptions.addBias = 1;
model_bs2 = ml_regression_basis(Xtrain, ytrain, options_bs2);
yhat_bs2 = model_bs2.predict(model_bs2, Xtest);
testError_bs2 = mean(abs(yhat_bs2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_bs2.name, testError_bs2);

%%
plotRegression1D(Xtrain, ytrain, model_l2, model_bs1, model_bs2);
title('Gaussian');