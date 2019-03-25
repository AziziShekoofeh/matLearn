%% Description of demo_regression_MLP
% Multi-layer perceptron regression with sigmoid and hyperbolic tangent
% activiation functions

clear all
close all
generateData_volatile

%% usage of MLP regression with sigmoid activation function
options_mlp1.nHidden= [5, 5];
options_mlp1.activFunc = 'sig';
[model_mlp1] = ml_regression_MLP(Xtrain,ytrain,options_mlp1);
yhat_mlp1 = model_mlp1.predict(model_mlp1, Xtest);
testError_mlp1 = mean(abs(yhat_mlp1 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_mlp1.name, testError_mlp1);

%% usage of MLP regression with tanh activation function

options_mlp2.nHidden= [8,8,8,8];
options_mlp2.activFunc = 'tanh';
[model_mlp2] = ml_regression_MLP(Xtrain,ytrain,options_mlp2);
yhat_mlp2 = model_mlp2.predict(model_mlp2, Xtest);
testError_mlp2 = mean(abs(yhat_mlp2 - ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n', model_mlp2.name, testError_mlp2);

%%
plotRegression1D(Xtrain, ytrain, model_mlp1, model_mlp2);
