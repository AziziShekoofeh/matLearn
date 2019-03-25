Xtrain = [];
ytrain = [];
for i = linspace(-1, 1, 12);
    for j = linspace(-1, 1, 12);
        Xtrain = [Xtrain; i + 0.5*(2*rand(1,1)-1) j + 0.5*(2*rand(1,1)-1)];
        if sqrt(abs(i)+abs(j)) < 1
            ytrain = [ytrain; -1];
        else
            ytrain = [ytrain; 1];
        end
    end
end

Xtest = [];
ytest = [];
for i = linspace(-1, 1, 15);
    for j = linspace(-1, 1, 15);
        Xtest = [Xtest; i j];
        if sqrt(abs(i)+abs(j)) < 1
            ytest = [ytest; -1];
        else
            ytest = [ytest; 1];
        end
    end
end