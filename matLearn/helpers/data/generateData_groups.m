  Xtrain = [];
   for i = linspace(-1, 1, 12);
       for j = linspace(-1, 1, 12);
    Xtrain = [Xtrain; i + 0.5*(2*rand(1,1)-1) j + 0.5*(2*rand(1,1)-1)];
       end
   end

ytrain = [repmat([-ones(6,1);ones(6,1)],[6,1]);repmat([ones(6,1);-ones(6,1)],[6,1])];

  Xtest = [];
   for i = linspace(-1, 1, 15);
       for j = linspace(-1, 1, 15);
    Xtest = [Xtest; i j];
       end
   end
ytest = [repmat([-ones(7,1);ones(8,1)],[7,1]);repmat([ones(7,1);-ones(8,1)],[8,1])];
  
