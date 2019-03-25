Xtrain = [];
ytrain = [];
randArray = [];
for i = 1:10
    a = 5*(2*rand(1,1))-1;
    b = 5*(2*rand(1,1))-1;
    for j = 1:5
        Xtrain = [Xtrain; a + 0.2*(2*rand(1,1)-1) b + 0.2*(2*rand(1,1)-1)];
        ytrain = [ytrain; i];
    end
    randArray = [randArray; a b];
    a = (2*rand(1,1))-1;
    b = (2*rand(1,1))-1;
    for j = 1:5
        Xtrain = [Xtrain; a + 0.2*(2*rand(1,1)-1) b + 0.2*(2*rand(1,1)-1)];
        ytrain = [ytrain; i];
    end
    randArray = [randArray; a b];
end

Xtest = [];
ytest = [];
for i = 1:10
    a = randArray(2*i-1,1);
    b = randArray(2*i-1,2);
    for j = 1:5
        Xtest = [Xtest; a + 0.2*(2*rand(1,1)-1) b + 0.2*(2*rand(1,1)-1)];
        ytest = [ytest; i];
    end
    a = randArray(2*i,1);
    b = randArray(2*i,2);
    for j = 1:5
        Xtest = [Xtest; a + 0.2*(2*rand(1,1)-1) b + 0.2*(2*rand(1,1)-1)];
        ytest = [ytest; i];
    end
end