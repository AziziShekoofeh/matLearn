%% Description of demo_multiclass_KDE.m
% Demonstrates generative kernel density estimation with RBF and polynomial
% kernels, with a Gaussian maximum likelihood fit as baseline

clear all
close all
generateData_clustersXonly

%% usage of generative Gaussian model
options_gs = [];
model_gs = ml_generative_Gaussian(Xtrain, ytrain, options_gs);
figure;
plotPDF(Xtrain, model_gs);
title('Generative Gaussian Model');

%% usage of generative Mixture of Multivariate Exponential Power model
options_mep = [];
options_mep.nMixtures = 5;
options_mep.init = 'kpp';
options_mep.nRestarts = 20; % overcome EM algorithm's local minima problem
options_mep.kappa = 1; % Mixture of Laplacians is special case of MMEP
model_mep = ml_generative_mixtureMEP(Xtrain, ytrain, options_mep);
figure;
plotPDF(Xtrain, model_mep);
title('Generative MEP Mixture Model, \kappa = 1');

%% usage of generative Gaussian mixture model
options_gmm = [];
options_gmm.nMixtures = 5;
options_gmm.nRestarts = 10;
model_gmm = ml_generative_mixtureGaussian(Xtrain, ytrain, options_gmm);
figure;
plotPDF(Xtrain, model_gmm);
title('Generative Mixture Gaussian Model');

%% usage of generative RBF kernel density estimation model
options_kde = [];
options_kde.kernelOptions = struct('sigma',.75);
model_kde = ml_generative_KDE(Xtrain, ytrain, options_kde);
figure;
title('Generative RBF Kernel Density Estimation Model');
plotPDF(Xtrain, model_kde);

%% usage of generative polynomial kernel density estimation model
options_kde_poly = [];
options_kde_poly.kernelOptions = struct('order',3,'bias',1);
options_kde_poly.kernelOptions.kernelFunc = @ml_kernel_poly;
model_kde_poly = ml_generative_KDE(Xtrain, ytrain, options_kde_poly);
figure;
title('Generative Poly Kernel Density Estimation Model');
plotPDF(Xtrain, model_kde_poly);

