  Xtrain = [];
   for i = linspace(-1, 1, 12);
       for j = linspace(-1, 1, 12);
    Xtrain = [Xtrain; j + (2*rand(1,1)-1) i + (2*rand(1,1)-1)];
       end
   end

ytrain = [-ones(72,1);ones(72,1)];

  Xtest = [];
   for i = linspace(-1, 1, 15);
       for j = linspace(-1, 1, 15);
    Xtest = [Xtest; j i];
       end
   end
ytest = [-ones(112,1);ones(113,1)];
  
