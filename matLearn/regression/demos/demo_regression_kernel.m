%% Description of demo_regression_kernel
% Uses a variety of kernels for L2 regression on synthetic datasets

clear all
close all

%%
generateData_quad

%% usage of linear kernalized L2 regression (quad data)
options_kn1 = [];
options_kn1.addBias = 1;
options_kn1.lambdaL2 = 1;
options_kn1.kernelFunc = @ml_kernel_gram;
options_kn1.kernelOptions = [];
model_kn1 = ml_regression_kernel(Xtrain, ytrain, options_kn1);
yhat_kn1 = model_kn1.predict(model_kn1, Xtest);
testError_kn1 = mean(abs(yhat_kn1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_kn1.name, testError_kn1);

%% usage of polynomial kernalized L2 regression (quad data)
options_kn2 = [];
options_kn2.addBias = 1;
options_kn2.lambdaL2 = 1;
options_kn2.kernelFunc = @ml_kernel_poly;
options_kn2.kernelOptions.bias = 0;
options_kn2.kernelOptions.order = 2;
model_kn2 = ml_regression_kernel(Xtrain, ytrain, options_kn2);
yhat_kn2 = model_kn2.predict(model_kn2, Xtest);
testError_kn2 = mean(abs(yhat_kn2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_kn2.name, testError_kn2);

%% usage of RBF kernalized L2 regression (quad data)
options_kn3 = [];
options_kn3.addBias = 1;
options_kn3.lambdaL2 = 1;
options_kn3.kernelFunc = @ml_kernel_rbf;
options_kn3.kernelOptions.sigma = 0.2;
model_kn3 = ml_regression_kernel(Xtrain, ytrain, options_kn3);
yhat_kn3 = model_kn3.predict(model_kn3, Xtest);
testError_kn3 = mean(abs(yhat_kn2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_kn3.name, testError_kn3);

%%
plotRegression1D(Xtrain, ytrain, model_kn1, model_kn2, model_kn3);

%%
generateData_sigmoid

%% usage of linear kernalized L2 regression (sigmoid data)
options_kn1 = [];
options_kn1.addBias = 1;
options_kn1.lambdaL2 = 1;
options_kn1.kernelFunc = @ml_kernel_gram;
options_kn1.kernelOptions = [];
model_kn1 = ml_regression_kernel(Xtrain, ytrain, options_kn1);
yhat_kn1 = model_kn1.predict(model_kn1, Xtest);
testError_kn1 = mean(abs(yhat_kn1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_kn1.name, testError_kn1);

%% usage of polynomial kernalized L2 regression (sigmoid data)
options_kn2 = [];
options_kn2.addBias = 1;
options_kn2.lambdaL2 = 1;
options_kn2.kernelFunc = @ml_kernel_poly;
options_kn2.kernelOptions.bias = 0;
options_kn2.kernelOptions.order = 3;
model_kn2 = ml_regression_kernel(Xtrain, ytrain, options_kn2);
yhat_kn2 = model_kn2.predict(model_kn2, Xtest);
testError_kn2 = mean(abs(yhat_kn2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_kn2.name, testError_kn2);

%% usage of RBF kernalized L2 regression (sigmoid data)
options_kn3 = [];
options_kn3.addBias = 1;
options_kn3.lambdaL2 = 1;
options_kn3.kernelFunc = @ml_kernel_rbf;
options_kn3.kernelOptions.sigma = 0.8;
model_kn3 = ml_regression_kernel(Xtrain, ytrain, options_kn3);
yhat_kn3 = model_kn3.predict(model_kn3, Xtest);
testError_kn3 = mean(abs(yhat_kn2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_kn3.name, testError_kn3);

%%
plotRegression1D(Xtrain, ytrain, model_kn1, model_kn2, model_kn3);

%%
generateData_Gauss

%% usage of linear kernalized L2 regression (Gauss data)
options_kn1 = [];
options_kn1.addBias = 1;
options_kn1.lambdaL2 = 1;
options_kn1.kernelFunc = @ml_kernel_gram;
options_kn1.kernelOptions = [];
model_kn1 = ml_regression_kernel(Xtrain, ytrain, options_kn1);
yhat_kn1 = model_kn1.predict(model_kn1, Xtest);
testError_kn1 = mean(abs(yhat_kn1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_kn1.name, testError_kn1);

%% usage of polynomial kernalized L2 regression (Gauss data)
options_kn2 = [];
options_kn2.addBias = 1;
options_kn2.lambdaL2 = 1;
options_kn2.kernelFunc = @ml_kernel_poly;
options_kn2.kernelOptions.bias = 0;
options_kn2.kernelOptions.order = 2;
model_kn2 = ml_regression_kernel(Xtrain, ytrain, options_kn2);
yhat_kn2 = model_kn2.predict(model_kn2, Xtest);
testError_kn2 = mean(abs(yhat_kn2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_kn2.name, testError_kn2);

%% usage of RBF kernalized L2 regression (Gauss data)
options_kn3 = [];
options_kn3.addBias = 1;
options_kn3.lambdaL2 = 1;
options_kn3.kernelFunc = @ml_kernel_rbf;
options_kn3.kernelOptions.sigma = 0.5;
model_kn3 = ml_regression_kernel(Xtrain, ytrain, options_kn3);
yhat_kn3 = model_kn3.predict(model_kn3, Xtest);
testError_kn3 = mean(abs(yhat_kn2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_kn3.name, testError_kn3);

%%
plotRegression1D(Xtrain, ytrain, model_kn1, model_kn2, model_kn3);