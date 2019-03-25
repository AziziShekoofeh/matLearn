%% Description demo_unsupervised_ISOMAP.m
% Demonstrates use of ISOMAP to visualize a dataset in lower dimensions
clear all
close all
load animals.mat
[n,d] = size(X);

%% usage of KPCA with rbf basis
% Reduce to 2-dimensions with KPCA
kernelArgs = struct('sigma',10);
options = struct('maxComponents',2,'kernelFunc',@ml_kernel_rbf,...
                 'kernelArgs',kernelArgs);
model = ml_unsupervised_dimRedKPCA(X,options);
Xreduced = model.reduceFunc(model,X);
plot(Xreduced(:,1),Xreduced(:,2),'.');
grid on
grid minor
title('KPCA Projection onto 2-dimensions of animals data (rbf kernel)');
% gname(animals)

%% usage ISOMAP to visualize animals dataset in low dimensions
options = [];
options.K = 2;
options.names = animals;
options.disconnected = 1;
ml_visualize_ISOMAP(X,options);
