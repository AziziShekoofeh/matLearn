%% Description of demo_multilabel_MLP.m
% Uses MLPs (Neural Networks) for multilabel classification with
% regularization on different layers and with different architectures.

clear all
close all
generateData_multiLabel

%% usage of multilabel MLP with two hidden layers
options = struct('nLabels',nLabels,...
                    'nHidden',[10 3]);
model = ml_multilabel_MLP(Xtrain,ytrain,options);
yhatTest = model.predict(model, Xtest);
yhatTrain = model.predict(model, Xtrain);
testError = sum(ytest~=yhatTest)/length(ytest);
model.trainError = sum(ytrain~=yhatTrain)/length(ytrain);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model.name, testError);
linear_makeOneContourPlot(Xtrain,ytrain, model);

%% usage of L2-regularized multilabel MLP with two hidden layers
options = struct('nLabels',nLabels,...
                 'lambdaO',1e-2,... % regularize output layer weights
                 'nHidden',[10 3]); 
model = ml_multilabel_MLP(Xtrain,ytrain,options);
yhatTest = model.predict(model, Xtest);
yhatTrain = model.predict(model, Xtrain);
testError = sum(ytest~=yhatTest)/length(ytest);
model.trainError = sum(ytrain~=yhatTrain)/length(ytrain);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model.name, testError);
linear_makeOneContourPlot(Xtrain,ytrain, model);

%% usage of MLP with three hidden layers
options = struct('nLabels',nLabels,...
                  'nHidden',[10 10 3]);
model = ml_multilabel_MLP(Xtrain,ytrain,options);
yhatTest = model.predict(model, Xtest);
yhatTrain = model.predict(model, Xtrain);
testError = sum(ytest~=yhatTest)/length(ytest);
model.trainError = sum(ytrain~=yhatTrain)/length(ytrain);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model.name, testError);
linear_makeOneContourPlot(Xtrain,ytrain, model);