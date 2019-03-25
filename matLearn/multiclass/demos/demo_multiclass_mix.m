%% Description of demo_multiclass_mix.m
% Fits a single generative Gaussian and a generative Gaussian mixture model 
% to a dataset

clear all
close all
generateData_multiXonly

%% usage of generative Gaussian model
options_gs = [];
model_gs = ml_generative_Gaussian(Xtrain, ytrain, options_gs);

%% usage of generative Gaussian mixture model
options_mg = [];
options_mg.nMixtures = 2;
model_mg =  ml_generative_mixtureGaussian(Xtrain, ytrain, options_mg);

%%
figure;
plotPDF(Xtrain, model_gs);
title('Generative Gaussian Model');
figure;
title('Generative Gaussian Mixture Model');
plotPDF(Xtrain, model_mg);
