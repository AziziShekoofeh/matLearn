Xtrain = [];
for i = 1:100
    Xtrain = [Xtrain; ([cos(pi/4), -sin(pi/4); sin(pi/4), ...
              cos(pi/4)]*[3*randn(1, 1);randn(1,1)])'];
end
ytrain = [];