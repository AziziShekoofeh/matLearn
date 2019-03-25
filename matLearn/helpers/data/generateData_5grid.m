Xtrain = [];
ytrain = [];
for i = linspace(-1, 1, 12);
    for j = linspace(-1, 1, 12);
        if (i < 1*j - 0.15 && -i < 0.7*j + 0.3);
            ytrain = [ytrain; 1];
        elseif (i >= 1*j - 0.15 && -i < 2*j);
            ytrain = [ytrain; 2];
        elseif (-i >= 0.7*j + 0.3 && i < 2.5*j - 0.3);
            ytrain = [ytrain; 3];
        elseif (i >= 2.5*j - 0.3 && i < 0.2*j);
            ytrain = [ytrain; 4];
        else
            ytrain = [ytrain; 6];
        end
        Xtrain = [Xtrain; i + 0.3*(2*rand(1,1)-1) j + 0.3*(2*rand(1,1)-1)];
    end
end

Xtest = [];
ytest = [];
for i = linspace(-1, 1, 15);
    for j = linspace(-1, 1, 15);
        if (i < 1*j - 0.15 && -i < 0.7*j + 0.3);
            ytest = [ytest; 1];
        elseif (i >= 1*j - 0.15 && -i < 2*j);
            ytest = [ytest; 2];
        elseif (-i >= 0.7*j + 0.3 && i < 2.5*j - 0.3);
            ytest = [ytest; 3];
        elseif (i >= 2.5*j - 0.3 && i < 0.2*j);
            ytest = [ytest; 4];
        else
            ytest = [ytest; 6];
        end
        Xtest = [Xtest; i j];
    end
end