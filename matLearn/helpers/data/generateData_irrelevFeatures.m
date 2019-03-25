Xtrain = [];
for n = linspace(-1, 1, 12);
	i = repmat(n, 12, 1) + rand(12, 1);
    j = repmat(n, 12, 1) + rand(12, 1);
	Xtrain = [Xtrain; i j];
end
ytrain = 0.5*randn(144,1) + Xtrain*ones(2,1);
Xtrain = [Xtrain, ones(size(Xtrain,1), 8)];

Xtest = [];
for i = linspace(-1, 1, 15);
    for j = linspace(-1, 1, 15);
        Xtest = [Xtest; i j];
    end
end
ytest = Xtest*ones(2,1);
Xtest = [Xtest, ones(size(Xtest,1), 8)];
  