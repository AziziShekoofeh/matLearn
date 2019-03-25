%% Description of demo_unsupervised_clustering.m
% Demonstrates two clustering methods: K-means and Density-Based clustering
% using the DBSCAN algorithm
clear all
close all
f=1; % figure index
generateData_clustering

%% Density-Based Clustering
options.eps = 2;
options.minPts = 2;
figure(f);f=f+1;
model = ml_unsupervised_DBcluster(Xtrain,options);
title('Density-Based clustering');

%% K-Means Clustering
options.k = 5;
options.kpp = 1; % Use K-means++ initialization
figure(f);f=f+1;
model = ml_unsupervised_clusterKmeans(Xtrain,options);
title('K-means Clustering with K-means++ Initialization');

