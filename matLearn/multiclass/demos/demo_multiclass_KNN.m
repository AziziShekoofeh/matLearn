%% Description of demo_multiclass_KNN.m
% Demonstrates multiclass KNN and multiclass logistic regression

clear all
close all
generateData_5grid

%% usage of k-nearest neighbours classification (5grid data)
options_knn = [];
options_knn.k = 5;
model_knn = ml_multiclass_KNN(Xtrain, ytrain, options_knn);
yhat_knn = model_knn.predict(model_knn, Xtest);
testError_st = mean(yhat_knn ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_knn.name, testError_st);

%% usage of multi-class logistic classification (5grid data)
options_lg = [];
options_lg.addBias = 1;
model_lg = ml_multiclass_logistic(Xtrain, ytrain, options_lg);
yhat_lg = model_lg.predict(model_lg, Xtest);
testError_lg = mean(yhat_lg ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_lg.name, testError_lg);

%%
figure;
plotClassifier(Xtrain, ytrain, model_knn);
figure;
plotClassifier(Xtrain, ytrain, model_lg);

generateData_gridMulti

%% usage of k-nearest neighbours classification (gridMulti data)
options_knn = [];
options_knn.k = 10;
model_knn = ml_multiclass_KNN(Xtrain, ytrain, options_knn);
yhat_knn = model_knn.predict(model_knn, Xtest);
testError_st = mean(yhat_knn ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_knn.name, testError_st);

%% usage of multi-class logistic classification (gridMulti data)
options_lg = [];
options_lg.addBias = 1;
model_lg = ml_multiclass_logistic(Xtrain, ytrain, options_lg);
yhat_lg = model_lg.predict(model_lg, Xtest);
testError_lg = mean(yhat_lg ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_lg.name, testError_lg);

%%
figure;
plotClassifier(Xtrain, ytrain, model_knn);
figure;
plotClassifier(Xtrain, ytrain, model_lg);
