Xtrain = linspace(-1, 1, 30)';
ytrain = 0.5*randn(30, 1) + Xtrain;
Xtrain=[Xtrain;5*rand(5,1)];
ytrain=[ytrain;0.5+3*randn(5,1)];
Xtest = linspace(-1, 1, 50)';
ytest = Xtest;