%% Description of demo_multiclass_decisions.m
% Shows the performance of stump and decision trees on a variety of
% different datasets

clear all
close all
generateData_binary

%% usage of stump classification (binary data)
options_st = [];
model_st = ml_multiclass_stump(Xtrain, ytrain, options_st);
yhat_st = model_st.predict(model_st, Xtest);
testError_st = mean(yhat_st ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_st.name, testError_st);

%% usage of decision tree classification (binary data)
options_dt = [];
model_dt = ml_multiclass_decisionTree(Xtrain, ytrain, options_dt);
yhat_dt = model_dt.predict(model_dt, Xtest);
testError_dt = mean(yhat_dt ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_dt.name, testError_dt);

%%
figure;
plotClassifier(Xtrain, ytrain, model_st);
figure;
plotClassifier(Xtrain, ytrain, model_dt);

generateData_4grid

%% usage of stump classification (4grid data)
options_st = [];
model_st = ml_multiclass_stump(Xtrain, ytrain, options_st);
yhat_st = model_st.predict(model_st, Xtest);
testError_st = mean(yhat_st ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_st.name, testError_st);

%% usage of decision tree classification (4grid data)
options_dt = [];
model_dt = ml_multiclass_decisionTree(Xtrain, ytrain, options_dt);
yhat_dt = model_dt.predict(model_dt, Xtest);
testError_dt = mean(yhat_dt ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_dt.name, testError_dt);

%%
figure;
plotClassifier(Xtrain, ytrain, model_st);
figure;
plotClassifier(Xtrain, ytrain, model_dt);

generateData_gridMulti

%% usage of stump classification (gridMulti data)
options_st = [];
model_st = ml_multiclass_stump(Xtrain, ytrain, options_st);
yhat_st = model_st.predict(model_st, Xtest);
testError_st = mean(yhat_st ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_st.name, testError_st);

%% usage of decision tree classification (gridMulti data)
options_dt = [];
model_dt = ml_multiclass_decisionTree(Xtrain, ytrain, options_dt);
yhat_dt = model_dt.predict(model_dt, Xtest);
testError_dt = mean(yhat_dt ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_dt.name, testError_dt);

%%
figure;
plotClassifier(Xtrain, ytrain, model_st);
figure;
plotClassifier(Xtrain, ytrain, model_dt);
