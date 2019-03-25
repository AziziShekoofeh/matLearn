%% Description of demo_multiclass_MLP.m
% Comparison of multiclass classification using multiclass logistic
% regression and multi-layer perceptron algorithms

clear all
close all
generateData_5grid

%% usage of multi-class logistic classification (5grid data)
options_lg = [];
options_lg.addBias = 1;
model_lg = ml_multiclass_logistic(Xtrain, ytrain, options_lg);
yhat_lg = model_lg.predict(model_lg, Xtest);
testError_lg = mean(yhat_lg ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', model_lg.name, testError_lg);

%% usage of MLP classification (5grid data)
options_mlp = [];
options_mlp.nHidden = [3 3 5];
model_mlp = ml_multiclass_MLP(Xtrain, ytrain, options_mlp);
yhat_mlp = model_mlp.predict(model_mlp, Xtest);
testError_mlp = mean(abs(yhat_mlp ~= ytest));
fprintf('Averaged misclassification test error with %s is: %.3f\n', model_mlp.name, testError_mlp);

%%
figure;
plotClassifier(Xtrain, ytrain, model_lg);
figure;
plotClassifier(Xtrain, ytrain, model_mlp);

generateData_gridMulti

%% usage of multi-class logistic classification (gridMulti data)
options_lg = [];
options_lg.addBias = 1;
model_lg = ml_multiclass_logistic(Xtrain, ytrain, options_lg);
yhat_lg = model_lg.predict(model_lg, Xtest);
testError_lg = mean(yhat_lg ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', model_lg.name, testError_lg);

%% usage of MLP classification (gridMulti data)
options_sp = [];
options_sp.nHidden = [5 5 5 5];
model_mlp = ml_multiclass_MLP(Xtrain, ytrain, options_sp);
xhat_sp = model_mlp.predict(model_mlp, Xtest);
testError_mlp = mean(xhat_sp ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', model_mlp.name, testError_mlp);

%%
figure;
plotClassifier(Xtrain, ytrain, model_lg);
figure;
plotClassifier(Xtrain, ytrain, model_mlp);
