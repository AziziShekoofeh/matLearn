%% Description of demo_multiclass_boosting.m
% Demonstrates boosted stump regression using the AdaBoost algorithm for
% a multiclass classification problem

close all
clear all
generateData_4grid

%% usage of bagged decision trees
options_bag = [];
options_bag.subModel = @ml_multiclass_stump;
model_bag = ml_multiclass_bagging(Xtrain, ytrain, options_bag);
yhat_bag = model_bag.predict(model_bag, Xtest);
testError_bag = mean(yhat_bag ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_bag.name, testError_bag);

%% usage of boosted multi-class stump regression with AdaBoost
options_bs1 = [];
options_bs1.nBoosts = 1000;
options_bs1.booster = 'ada';
options_bs1.subModel = @ml_multiclass_stump;
model_bs1 = ml_multiclass_boosting(Xtrain, ytrain, options_bs1);
yhat_bs1 = model_bs1.predict(model_bs1, Xtest);
testError_bs1 = mean(yhat_bs1 ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_bs1.name, testError_bs1)

%%
figure;
plotClassifier(Xtrain, ytrain, model_bag);
figure;
plotClassifier(Xtrain, ytrain, model_bs1);
