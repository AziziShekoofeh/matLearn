Xtrain = [];
ytrain = [];
for i = linspace(-1, 1, 12);
    for j = linspace(-1, 1, 12);
        if (j < 0 && i > 0);
            ytrain = [ytrain; 1];
        elseif (j >= 0 && i <= 0);
            ytrain = [ytrain; 2];
        elseif (j < 0 && i <= 0);
            ytrain = [ytrain; 3];
        else
            ytrain = [ytrain; 6];
        end
        Xtrain = [Xtrain; i + 0.5*(2*rand(1,1)-1) j + 0.5*(2*rand(1,1)-1)];
    end
end

Xtest = [];
ytest = [];
for i = linspace(-1, 1, 15);
    for j = linspace(-1, 1, 15);
        if (j < 0 && i > 0);
            ytest= [ytest; 1];
        elseif (j >= 0 && i <= 0);
            ytest = [ytest; 2];
        elseif (j < 0 && i <= 0);
            ytest = [ytest; 3];
        else
            ytest = [ytest; 6];
        end
        Xtest = [Xtest; i j];
    end
end