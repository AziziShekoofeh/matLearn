%% Description of demo_multilabel_multinomial
% Multilabel classification using multinomial logistic regression with and
% without L2 regularization

clear all
close all
generateData_multiLabel

%% usage of multilabel multinomial logistic regression
options = struct('nLabels',nLabels,...
                 'nClasses',max(ytrain)+1);
model = ml_multilabel_multinomial(Xtrain,ytrain,options);
yhatTest = model.predict(model, Xtest);
yhatTrain = model.predict(model, Xtrain);
testError = sum(ytest~=yhatTest)/length(ytest);
model.trainError = sum(ytrain~=yhatTrain)/length(ytrain);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model.name, testError);
linear_makeOneContourPlot(Xtrain,ytrain, model)

%% usage of multilabel L2-regularized multinomial logistic regression
options = struct('nLabels',nLabels,...
                 'nClasses',max(ytrain)+1,...
                 'lambdaL2',1e-2);
model = ml_multilabel_multinomial(Xtrain,ytrain,options);
yhatTest = model.predict(model, Xtest);
yhatTrain = model.predict(model, Xtrain);
testError = sum(ytest~=yhatTest)/length(ytest);
model.trainError = sum(ytrain~=yhatTrain)/length(ytrain);
fprintf('Averaged misclassification test error with %s is: %.3f\n',...
        model.name, testError);
linear_makeOneContourPlot(Xtrain,ytrain, model)
