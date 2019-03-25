%% Description of demo_binaryclass_decisions.m
% Demonstrates stump, tree, and forest binary classification on three
% different datasets
clear all
close all
generateData_vert

%% usage of stump binary classification (vert data)
options_st = [];
model_st = ml_binaryclass_stump(Xtrain, ytrain, options_st);
yhat_st = model_st.predict(model_st, Xtest);
testError_st = mean(yhat_st ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_st.name, testError_st);

%% usage of tree binary classification (vert data)
options_tr = [];
options_tr.dataTypes = [1 1];
options_tr.maxDepth = 16;
model_tr = ml_binaryclass_tree(Xtrain, ytrain, options_tr);
yhat_tr = model_tr.predict(model_tr, Xtest);
testError_tr = mean(yhat_tr ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_tr.name, testError_tr);

%% usage of random forest binary classification (vert data)
options_rf = [];
options_rf.oobMaxFeatures = [1 2];
model_rf = ml_binaryclass_randomForest(Xtrain, ytrain, options_rf);
yhat_rf = model_rf.predict(model_rf, Xtest);
testError_rf = mean(yhat_rf ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_rf.name, testError_rf);

%%
figure;
plot2DClassifier(Xtrain, ytrain, model_st);
figure;
plot2DClassifier(Xtrain, ytrain, model_tr);
figure;
plot2DClassifier(Xtrain, ytrain, model_rf);

generateData_slanted

%% usage of stump binary classification (slanted data)
options_st = [];
model_st = ml_binaryclass_stump(Xtrain, ytrain, options_st);
yhat_st = model_st.predict(model_st, Xtest);
testError_st = mean(abs(yhat_st - ytest));
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_st.name, testError_st);

%% usage of tree binary classification (slanted data)
options_tr = [];
options_tr.dataTypes = [1 1];
options_tr.maxDepth = 16;
model_tr = ml_binaryclass_tree(Xtrain, ytrain, options_tr);
yhat_tr = model_tr.predict(model_tr, Xtest);
testError_tr = mean(yhat_tr ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_tr.name, testError_tr);

%% usage of random forest binary classification (slanted data)
options_rf = [];
options_rf.oobMaxFeatures = [1 2];
model_rf = ml_binaryclass_randomForest(Xtrain, ytrain, options_rf);
yhat_rf = model_rf.predict(model_rf, Xtest);
testError_rf = mean(yhat_rf ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_rf.name, testError_rf);

%%
figure;
plot2DClassifier(Xtrain, ytrain, model_st);
figure;
plot2DClassifier(Xtrain, ytrain, model_tr);
figure;
plot2DClassifier(Xtrain, ytrain, model_rf);

generateData_groups

%% usage of stump binary classification (groups data)
options_st = [];
model_st = ml_binaryclass_stump(Xtrain, ytrain, options_st);
yhat_st = model_st.predict(model_st, Xtest);
testError_st = mean(yhat_st ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_st.name, testError_st);

%% usage of tree binary classification (groups data)
options_tr = [];
options_tr.dataTypes = [1 1];
model_tr = ml_binaryclass_tree(Xtrain, ytrain, options_tr);
yhat_tr = model_tr.predict(model_tr, Xtest);
testError_tr = mean(yhat_tr ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_tr.name, testError_tr);

%% usage of random forest binary classification (groups data)
options_rf = [];
options_rf.oobMaxFeatures = [1 2];
model_rf = ml_binaryclass_randomForest(Xtrain, ytrain, options_rf);
yhat_rf = model_rf.predict(model_rf, Xtest);
testError_rf = mean(yhat_rf ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_rf.name, testError_rf);

%%
figure;
plot2DClassifier(Xtrain, ytrain, model_st);
figure;
plot2DClassifier(Xtrain, ytrain, model_tr);
figure;
plot2DClassifier(Xtrain, ytrain, model_rf);