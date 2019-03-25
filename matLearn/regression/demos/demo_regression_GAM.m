%% Description of demo_regression_GAM.m
% Regression using General Additive Models 

clear all
close all
generateData_3DQuad

%% usage of GAM regression with linear regression
options_gam1 = [];
options_gam1.subFunc = 'lin';
model_gam1 = ml_regression_GAM(Xtrain, ytrain, options_gam1);
yhat_gam1 = model_gam1.predict(model_gam1, Xtest);
testError_gam1 = mean(abs(yhat_gam1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_gam1.name, testError_gam1);
plotRegression2DPoints(Xtrain, ytrain, model_gam1);
view(60,30);

%% usage of GAM regression with polynomial of degree 2
options_gam2 = [];
options_gam2.subFunc = 'rg';
options_gam2.deg = 2;
model_gam2 = ml_regression_GAM(Xtrain, ytrain, options_gam2);
yhat_gam2 = model_gam2.predict(model_gam2, Xtest);
testError_gam2 = mean(abs(yhat_gam2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_gam2.name, testError_gam2);
plotRegression2DPoints(Xtrain, ytrain, model_gam2);
view(60, 30);

%% usage of GAM regression with smooth cubic splines
options_gam3 = [];
options_gam3.subFunc = 'spl';
options_gam3.smoothing = 1.5;
model_gam3 = ml_regression_GAM(Xtrain, ytrain, options_gam3);
yhat_gam3 = model_gam3.predict(model_gam3, Xtest);
testError_gam3 = mean(abs(yhat_gam3 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_gam3.name, testError_gam3);
plotRegression2DPoints(Xtrain, ytrain, model_gam3);
legend
view(-60, 30);


