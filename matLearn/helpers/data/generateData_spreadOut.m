Xtrain1 = linspace(-1, 1, 10)';
ytrain1 = 5*randn(10, 1) + Xtrain1;

Xtrain2 = linspace(-1, 1, 90)';
ytrain2 = 0.3*randn(90, 1) + Xtrain2;

Xtrain=[Xtrain1;Xtrain2];
ytrain=[ytrain1;ytrain2];

Xtest = linspace(-1, 1, 100)';
ytest = Xtest;