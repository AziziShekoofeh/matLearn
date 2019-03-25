Xtrain = [];
ytrain = [];
for i = [1:4,6]
    a = 5*((2*rand(1,1))-1);
    b = 5*((2*rand(1,1))-1);
    for j = 1:50
        Xtrain = [Xtrain; a + 0.5*(2*rand(1,1)-1) b + 0.5*(2*rand(1,1)-1)];
    end
end