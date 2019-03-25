%% Description of demo_multiclass_NB.m
% Finds maximum likelihood Naive Bayes and Gaussian distributions and plots
% the level curves of the resulting PDFs

clear all
close all
generateData_ovalXonly

%% usage of generative Naive Bayes model
options_nb = [];
model_nb = ml_generative_NB(Xtrain, ytrain, options_nb);

%% usage of generative Gaussian model
options_gs = [];
model_gs = ml_generative_Gaussian(Xtrain, ytrain, options_gs);

%%
figure;
plotPDF(Xtrain, model_nb);
title('Generative Naive Bayes Model');
figure;
plotPDF(Xtrain, model_gs);
title('Generative Gaussian Model');
