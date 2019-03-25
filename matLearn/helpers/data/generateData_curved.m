Xtrain = [];
for i = linspace(-1, 1, 12);
    for j = linspace(-1, 1, 12);
        Xtrain = [Xtrain; i + (2*rand(1,1)-1) j + (2*rand(1,1)-1)];
    end
end
ytrain = zeros(size(Xtrain,1),1);
index = Xtrain(:,1) <= -0.5.*(Xtrain(:,2)+1).*(Xtrain(:,2)-1);
ytrain(index) = -1;
ytrain(~index) = 1;

Xtest = [];
for i = linspace(-1, 1, 15);
    for j = linspace(-1, 1, 15);
        Xtest = [Xtest; i j];
    end
end
ytest = zeros(size(Xtest,1),1);
index = Xtest(:,1) <= -0.5.*(Xtest(:,2)+1).*(Xtest(:,2)-1);
ytest(index) = -1;
ytest(~index) = 1;