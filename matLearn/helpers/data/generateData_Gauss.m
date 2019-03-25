Xtrain = linspace(-1, 1, 30)';
ytrain = 0.4*randn(30,1)+10*exp(-5*Xtrain.^2);
Xtest = linspace(-1, 1, 50)';
ytest = 10*exp(-5*Xtest.^2);
