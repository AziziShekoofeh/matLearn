clear

Xtrain = [];
for i = -1:(1/6):1
    for j = i:(1/6):1
        Xtrain = [Xtrain; i+0.5*rand(1,1), j+0.5*rand(1,1)];
    end
end
ytrain = 0.5*rand(size(Xtrain,1),1) + Xtrain.^2*ones(2,1);

Xtest = [];
for i = -1:(2/15):1
    for j = i:(2/15):1
        Xtest = [Xtest; i j];
    end
end
ytest = Xtest.^2*ones(2,1);

