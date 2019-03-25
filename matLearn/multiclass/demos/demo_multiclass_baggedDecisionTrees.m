%% Description of demo_multiclass_baggedDecisionTrees.m
% Demonstrates bagged decision trees versus stump and decision trees for a
% multiclass classification task.

clear all
close all
generateData_4grid

%% stump classification baseline
options_st = [];
options_st.error = 'err';
model_st = ml_multiclass_stump(Xtrain, ytrain, options_st);
yhat_st = model_st.predict(model_st, Xtest);
testError_st = mean(yhat_st ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', model_st.name, testError_st);

%% decision tree classification baseline
options_dt = [];
model_dt = ml_multiclass_decisionTree(Xtrain, ytrain, options_dt);
yhat_dt = model_dt.predict(model_dt, Xtest);
testError_dt = mean(yhat_dt ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', model_dt.name, testError_dt);

%% decision tree classification with bagging
options_bg = [];
options_bg.nModels = 15;
options_bg.subModel = @ml_multiclass_decisionTree;
model_bg = ml_multiclass_bagging(Xtrain, ytrain, options_bg);
yhat_bg = model_bg.predict(model_bg, Xtest);
testError_bg = mean(yhat_bg ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model_bg.name, testError_bg)

%%
figure;
plotClassifier(Xtrain, ytrain, model_st);
figure;
plotClassifier(Xtrain, ytrain, model_dt);
figure;
plotClassifier(Xtrain, ytrain, model_bg);