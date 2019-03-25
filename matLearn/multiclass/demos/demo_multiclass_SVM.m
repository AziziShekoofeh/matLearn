%% Description demo_multiclass_SVM.m
% Demonstrates multiclass classification using SVM with two methods of 
% formulating the constrained optimization problem, N-slack and NK-slack

clear all
close all
generateData_5grid

%% usage of  N-slack SVM classification
options_svm1 = [];
options_svm1.addBias = 1;
options_svm1.slack = 'n';
model_svm1 = ml_multiclass_SVM(Xtrain, ytrain, options_svm1);
yhat_svm1 = model_svm1.predict(model_svm1, Xtest);
testError_svm1 = mean(yhat_svm1 ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_svm1.name, testError_svm1);

%% usage of NK-slack SVM classification
options_svm2 = [];
options_svm2.addBias = 1;
options_svm2.slack = 'nk';
model_svm2 = ml_multiclass_SVM(Xtrain, ytrain, options_svm2);
yhat_svm2 = model_svm2.predict(model_svm2, Xtest);
testError_svm2 = mean(yhat_svm2 ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_svm2.name, testError_svm2);

%%
figure;
plotClassifier(Xtrain, ytrain, model_svm1);
figure;
plotClassifier(Xtrain, ytrain, model_svm2);
