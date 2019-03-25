Xtrain = [];
ytrain = [];
randArray = [];
for i = [1:4,6]
    a = 5*((2*rand(1,1))-1);
    b = 5*((2*rand(1,1))-1);
    for j = 1:50
        Xtrain = [Xtrain; a + 0.5*(2*rand(1,1)-1) b + 0.5*(2*rand(1,1)-1)];
        ytrain = [ytrain; i];
    end
    randArray = [randArray; a b];
end

Xtest = [];
ytest = [];

counter = 1;
for i = [1:4,6]
    a = randArray(counter,1);
    b = randArray(counter,2);
    for j = 1:10
        Xtest = [Xtest; a + 0.2*(2*rand(1,1)-1) b + 0.2*(2*rand(1,1)-1)];
        ytest = [ytest; i];
    end
    counter = counter + 1;
end