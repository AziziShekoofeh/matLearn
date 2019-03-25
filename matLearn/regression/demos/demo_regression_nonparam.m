%% Description of demo_regression_nonparam
% Demonstrates nonparametric regression including KNN regression, 
% Nadaraya-Watson kernel regression, and local regression using L2 loss

clear all
close all
generateData_sigmoid

%% usage of KNN regression (sigmoid data)
options_knn = [];
options_knn.k = 10;
model_knn = ml_regression_KNN(Xtrain, ytrain, options_knn);
yhat_knn = model_knn.predict(model_knn, Xtest);
testError_knn = mean(abs(yhat_knn - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_knn.name, testError_knn);

%% usage of NW regression with RBF kernel (sigmoid data)
options_nw = [];
options_nw.kernelFunc = @ml_kernel_rbf;
options_nw.kernelOptions.sigma = 1;
model_nw = ml_regression_NW(Xtrain, ytrain, options_nw);
yhat_nw = model_nw.predict(model_nw, Xtest);
testError_nw = mean(abs(yhat_nw - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_nw.name, testError_nw);

%% usage of local L2 regression (sigmoid data)
options_lc = [];
options_lc.k = 15;
options_lc.subModel = @ml_regression_L2;
options_lc.subOptions.addBias = 1;
model_lc = ml_regression_local(Xtrain, ytrain, options_lc);
yhat_lc = model_lc.predict(model_lc, Xtest);
testError_lc = mean(abs(yhat_lc - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_lc.name, testError_lc);

%%
plotRegression1D(Xtrain, ytrain, model_knn, model_nw, model_lc);
title('Sigmoid');

generateData_Gauss

%% usage of KNN regression (Gauss data)
options_knn = [];
options_knn.k = 5;
model_knn = ml_regression_KNN(Xtrain, ytrain, options_knn);
yhat_knn = model_knn.predict(model_knn, Xtest);
testError_knn = mean(abs(yhat_knn - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_knn.name, testError_knn);

%% usage of NW regression with RBF kernel (Gauss data)
options_nw = [];
options_nw.kernelFunc = @ml_kernel_rbf;
options_nw.kernelOptions.sigma = 0.1;
model_nw = ml_regression_NW(Xtrain, ytrain, options_nw);
yhat_nw = model_nw.predict(model_nw, Xtest);
testError_nw = mean(abs(yhat_nw - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_nw.name, testError_nw);

%% usage of local L2 regression (Gauss data)
options_lc = [];
options_lc.k = 8;
options_lc.subModel = @ml_regression_L2;
options_lc.subOptions.addBias = 1;
model_lc = ml_regression_local(Xtrain, ytrain, options_lc);
yhat_lc = model_lc.predict(model_lc, Xtest);
testError_lc = mean(abs(yhat_lc - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_lc.name, testError_lc);

%%
plotRegression1D(Xtrain, ytrain, model_knn, model_nw, model_lc);
title('Gaussian');

generateData_quad

%% usage of KNN regression (quad data)
options_knn = [];
options_knn.k = 5;
model_knn = ml_regression_KNN(Xtrain, ytrain, options_knn);
yhat_knn = model_knn.predict(model_knn, Xtest);
testError_knn = mean(abs(yhat_knn - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_knn.name, testError_knn);

%% usage of NW regression with RBF kernel (quad data)
options_nw = [];
options_nw.kernelFunc = @ml_kernel_rbf;
options_nw.kernelOptions.sigma = 0.12;
model_nw = ml_regression_NW(Xtrain, ytrain, options_nw);
yhat_nw = model_nw.predict(model_nw, Xtest);
testError_nw = mean(abs(yhat_nw - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_nw.name, testError_nw);

%% usage of local L2 regression (quad data)
options_lc = [];
options_lc.k = 10;
options_lc.subModel = @ml_regression_L2;
options_lc.subOptions.addBias = 1;
model_lc = ml_regression_local(Xtrain, ytrain, options_lc);
yhat_lc = model_lc.predict(model_lc, Xtest);
testError_lc = mean(abs(yhat_lc - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', ...
        model_lc.name, testError_lc);

%%
plotRegression1D(Xtrain, ytrain, model_knn, model_nw, model_lc);
title('Quadratic');