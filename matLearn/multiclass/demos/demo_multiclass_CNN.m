%% Description of demo_multiclass_CNN.m
%
% Comparison of multiclass classification using multiclass logistic
% regression and simplest possible CNN with one convolution and one mean
% pooling layer feeding into softmax
%
% Note: implementations have not yet been parallelized and
% do not make use of GPUs in order to keep the algorithm easy to
% understand and extend. Consequently, this demo will take significant time
% to run.

% load data from MNIST
loadMNISTDataset

%% usage of multi-class logistic classification (MNIST data)
options_lg = [];

options_lg.addBias = 1;
model_lg = ml_multiclass_logistic(Xtrain, ytrain, options_lg);
yhat_lg = model_lg.predict(model_lg, Xtest);
testError_lg = mean(yhat_lg ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_lg.name, testError_lg);

%% usage of multi-class CNN classification (MNIST data)
options_cnn.imageDim = 28;
options_cnn.nClasses = 10; 
options_cnn.filterDim = 9;  % Filter size for conv layer
options_cnn.nFilters = 20;   % Number of filters for conv layer
options_cnn.poolDim = 2; 
model_cnn = ml_multiclass_CNN(Xtrain, ytrain, options_cnn);
yhat_cnn = model_cnn.predict(model_cnn, Xtest);
testError_cnn = mean(yhat_cnn ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_cnn.name, testError_cnn);