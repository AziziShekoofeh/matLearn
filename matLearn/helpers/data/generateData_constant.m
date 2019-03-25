Xtrain = [linspace(-1, 1, 30)'; linspace(-1, 1, 5)'];
ytrain = [0.2*randn(30, 1); 3*rand(5,1)] + mean(Xtrain)*ones(35,1);
Xtest = linspace(-1, 1, 50)';
ytest = Xtest;