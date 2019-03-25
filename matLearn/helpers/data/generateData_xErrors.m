% Xtrain = repmat(linspace(-1, 1, 8)', 2, 1) + rand(16, 1);
% ytrain = rand(16, 1) + Xtrain;
% Xtest = linspace(-1, 1, 15)';
% ytest = Xtest;

Xtrain = linspace(-1, 1, 12)';
ytrain = 0.5*randn(12, 1) + Xtrain;
Xtrain = 0.5*randn(12, 1) + Xtrain;
Xtest = linspace(-1, 1, 15)';
ytest = Xtest;