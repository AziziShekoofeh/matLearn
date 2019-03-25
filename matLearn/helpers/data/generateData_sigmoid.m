Xtrain = linspace(-5, 5, 30)';
ytrain = 0.05*randn(30,1)+Xtrain./(1+abs(Xtrain));
Xtest = linspace(-5, 5, 50)';
ytest = Xtest./(1+abs(Xtest));
