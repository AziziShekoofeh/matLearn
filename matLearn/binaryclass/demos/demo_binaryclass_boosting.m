%% Description of demo_binaryclass_boosting.m
% Comparison of AdaBoost and LogitBoost with stump classification as
% baseline on two datasets

close all
clear all
generateData_curved

%% usage of stump binary classification (curved data)
options_st = [];
model_st = ml_binaryclass_stump(Xtrain, ytrain, options_st);
yhat_st = model_st.predict(model_st, Xtest);
testError_st = mean(yhat_st ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_st.name, testError_st);

%% usage of boosting binary classification with AdaBoost (curved data)
options_bs1 = [];
options_bs1.nBoosts = 50;
options_bs1.booster = 'ada';
options_bs1.subModel = @ml_binaryclass_stump;
model_bs1 = ml_binaryclass_boosting(Xtrain, ytrain, options_bs1);
yhat_bs1 = model_bs1.predict(model_bs1, Xtest);
testError_bs1 = mean(yhat_bs1 ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_bs1.name, testError_bs1)

%% usage of boosting binary classification with LogitBoost (curved data)
options_bs2 = [];
options_bs2.nBoosts = 50;
options_bs2.booster = 'logit';
options_bs2.subModel = @ml_binaryclass_stump;
options_bs2.subOptions.addBias = 1;
model_bs2 = ml_binaryclass_boosting(Xtrain, ytrain, options_bs2);
yhat_bs2 = model_bs2.predict(model_bs2, Xtest);
testError_bs2 = mean(yhat_bs2 ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_bs2.name, testError_bs2)

%%
figure;
plot2DClassifier(Xtrain, ytrain, model_st);

figure;
plot2DClassifier(Xtrain, ytrain, model_bs1);
hold on;
for k = 1:length(model_bs1.trainModels);
    plot2DLine(model_bs1.trainModels{k});
    alpha(0)
end
plot2DClassifier_red(Xtrain, ytrain, model_bs1);

figure;
plot2DClassifier(Xtrain, ytrain, model_bs2);
hold on;
for k = 1:length(model_bs2.trainModels);
    plot2DLine(model_bs2.trainModels{k});
    alpha(0)
end
plot2DClassifier_red(Xtrain, ytrain, model_bs2);

generateData_slanted

%% usage of stump binary classification (slanted data)
options_st = [];
model_st = ml_binaryclass_stump(Xtrain, ytrain, options_st);
yhat_st = model_st.predict(model_st, Xtest);
testError_st = mean(yhat_st ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_st.name, testError_st);

%% usage of boosting binary classification with AdaBoost (slanted data)
options_bs1 = [];
options_bs1.nBoosts = 50;
options_bs1.booster = 'ada';
options_bs1.subModel = @ml_binaryclass_stump;
options_bs1.subOptions.addBias = 1;
model_bs1 = ml_binaryclass_boosting(Xtrain, ytrain, options_bs1);
yhat_bs1 = model_bs1.predict(model_bs1, Xtest);
testError_bs1 = mean(yhat_bs1 ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_bs1.name, testError_bs1)

%% usage of boosting binary classification with LogitBoost (slanted data)
options_bs2 = [];
options_bs2.nBoosts = 50;
options_bs2.booster = 'logit';
options_bs2.subModel = @ml_binaryclass_stump;
options_bs2.subOptions.addBias = 1;
model_bs2 = ml_binaryclass_boosting(Xtrain, ytrain, options_bs2);
yhat_bs2 = model_bs2.predict(model_bs2, Xtest);
testError_bs2 = mean(yhat_bs2 ~= ytest);
fprintf('Averaged misclassification test error with %s is: %.3f\n', ...
        model_bs2.name, testError_bs2)

%%
figure;
plot2DClassifier(Xtrain, ytrain, model_st);

figure;
plot2DClassifier(Xtrain, ytrain, model_bs1);
hold on;
for k = 1:length(model_bs1.trainModels);
    plot2DLine(model_bs1.trainModels{k});
    alpha(0)
end
plot2DClassifier_red(Xtrain, ytrain, model_bs1);

figure;
plot2DClassifier(Xtrain, ytrain, model_bs2);
hold on;
for k = 1:length(model_bs2.trainModels);
    plot2DLine(model_bs2.trainModels{k});
    alpha(0)
end
plot2DClassifier_red(Xtrain, ytrain, model_bs2);