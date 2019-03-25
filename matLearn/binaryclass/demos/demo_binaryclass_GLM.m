%% Description of demo_binaryclass_GLM.m
% Uses various link functions in a Generalized Linear Model for binary
% classification

clear all
close all
generateData_vert

%% usage of logistic regression
options_lg = [];
options_lg.addBias = 1;
model_lg = ml_binaryclass_logistic(Xtrain, ytrain, options_lg);
yhat_lg = model_lg.predict(model_lg, Xtest);
testError_lg = mean(yhat_lg ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_lg.name, testError_lg);
figure;
plot2DClassifier(Xtrain, ytrain, model_lg);

%% usage of probit loss binary classification
options_pb = [];
options_pb.addBias = 1;
options_pb.lambdaL2 = 1e-4;
model_pb = ml_binaryclass_probit(Xtrain, ytrain, options_pb);
yhat_pb = model_pb.predict(model_pb, Xtest);
testError_pb = mean(yhat_pb ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_pb.name, testError_pb);
figure;
plot2DClassifier(Xtrain, ytrain, model_pb);

%% usage of Cauchit loss binary classification
options_cc = [];
model_cc = ml_binaryclass_Cauchit(Xtrain, ytrain, options_cc);
yhat_cc = model_cc.predict(model_cc, Xtest);
testError_cc = mean(yhat_cc ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_cc.name, testError_cc);
figure;
plot2DClassifier(Xtrain, ytrain, model_cc);

%% usage of extreme loss binary classification
options_ex = [];
model_ex = ml_binaryclass_extreme(Xtrain, ytrain, options_ex);
yhat_ex = model_ex.predict(model_ex, Xtest);
testError_ex = mean(yhat_ex ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_ex.name, testError_ex);
figure;
plot2DClassifier(Xtrain, ytrain, model_ex);