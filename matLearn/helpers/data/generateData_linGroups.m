Xtrain1 = linspace(-1, 0, 25)';
ytrain1 = 0.1*randn(25, 1) + Xtrain1;

Xtrain2 = linspace(0, 1, 25)';
ytrain2 = 0.1*randn(25, 1) - Xtrain2;

Xtrain=[Xtrain1;Xtrain2];
ytrain=[ytrain1;ytrain2];

Xtest = linspace(-1, 1, 80)';
ytest = Xtest;