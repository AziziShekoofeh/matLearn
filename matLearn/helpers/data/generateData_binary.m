Xtrain = [];
ytrain = [];
for i = linspace(-1, 1, 12);
    for j = linspace(-1, 1, 12);
        if (j < 0);
            ytrain = [ytrain; 1];
        else
            ytrain = [ytrain; 2];
        end
        Xtrain = [Xtrain; i + 0.4*(2*rand(1,1)-1) j + 0.4*(2*rand(1,1)-1)];
    end
end

Xtest = [];
ytest = [];
for i = linspace(-1, 1, 15);
    for j = linspace(-1, 1, 15);
        if (j < 0);
            ytest= [ytest; 1];
        else
            ytest = [ytest; 2];
        end
        Xtest = [Xtest; i j];
    end
end