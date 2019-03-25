nFeatures = 5;
Xtrain = repmat([linspace(-1, 1, 30)'; linspace(-1, 1, 5)'],1, nFeatures);
ytrain = [0.2*randn(30, nFeatures); 3*rand(5,nFeatures)] + Xtrain;
Xtest = repmat(linspace(-1, 1, 50)',;
ytest = Xtest;