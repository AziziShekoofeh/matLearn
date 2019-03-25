Xtrain1 = linspace(-1, 0, 25)';
ytrain1 = 0.1*randn(25, 1) + mean(Xtrain1)*ones(25,1);

Xtrain2 = linspace(0, 1, 25)';
ytrain2 = 0.1*randn(25, 1) + mean(Xtrain2)*ones(25,1);

Xtrain=[Xtrain1;Xtrain2];
ytrain=[ytrain1;ytrain2];

Xtest = linspace(-1, 1, 80)';
ytest = Xtest;