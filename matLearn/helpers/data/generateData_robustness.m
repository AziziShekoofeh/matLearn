  Xtrain = [];
   for i = linspace(-1, 1, 12);
       for j = linspace(-1, 1, 12);
            Xtrain = [Xtrain; i + 0.75*rand(1,1) j + 0.75*rand(1,1)];
       end
   end

ytrain = [ones(5,1);-ones(67,1);ones(72,1)];

Xtest = [];
for i = linspace(-1, 1, 15);
    for j = linspace(-1, 1, 15);
        Xtest = [Xtest; i j];
    end
end
ytest = [-ones(112,1);ones(113,1)];
  
