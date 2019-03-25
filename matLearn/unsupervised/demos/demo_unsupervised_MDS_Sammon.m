%% Description of demo_unsupervised_MDS_Sammon.m
% Demonstrates classic Multi-Dimensional Scaling and Sammon mapping
clear all
close all
load cities.mat
Xtrain = ratings;

%% usage of MDS
% MDS projection to 2D
options = struct('nComponents',2);
Xreduced = ml_visualize_MDS(Xtrain,options);
plot(Xreduced(:,1),Xreduced(:,2),'.');
grid on
grid minor
title('MDS 2-dimensional visualization of cities data');
fprintf('Click plot to name cities, press any key to continue\n');
gname(names)

% 3D projection of dataset using MDS
options = struct('nComponents',3);
Xreduced = ml_visualize_MDS(Xtrain,options);
plot3(Xreduced(:,1),Xreduced(:,2),Xreduced(:,3),'.');
grid on
grid minor

%% usage of Sammon mapping
% Sammon mapping projection to 2D
options = struct('nComponents',2);
Xreduced = ml_visualize_Sammon(Xtrain,options);
plot(Xreduced(:,1),Xreduced(:,2),'.');
grid on
grid minor
title('Sammon mapping 2-dimensional visualization of cities data');
fprintf('Click plot to name cities, press any key to continue\n');
gname(names)

% Sammon mapping projection to 3D
options = struct('nComponents',3);
Xreduced = ml_visualize_Sammon(Xtrain,options);
plot3(Xreduced(:,1),Xreduced(:,2),Xreduced(:,3),'.');
grid on
grid minor