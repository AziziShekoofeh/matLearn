%% Description of demo_multiclass_student.m
% Fit generative Gaussian and Student's t models to a dataset and plot the
% results

clear all
close all
generateData_outliersXonly

%% usage of generative Gaussian model
options_gs = [];
model_gs = ml_generative_Gaussian(Xtrain, ytrain, options_gs);

%% usage of generative Student-t model
options_st = [];
model_st = ml_generative_student(Xtrain, ytrain, options_st);

%%
figure;
plotPDF(Xtrain, model_gs);
title('Generative Gaussian Model');
figure;
plotPDF(Xtrain, model_st);
title('Generative Student''s-t Model');
