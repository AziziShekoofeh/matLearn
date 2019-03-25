  Xtrain = [];
  ytrain = [];
   for i = linspace(-1, 1, 12);
       for j = linspace(-1, 1, 12);
           if (i < 0);
               ytrain = [ytrain; -1];
           else
               if (rand(1,1) <= 0.5);
                   ytrain = [ytrain; -1];
               else
                   ytrain = [ytrain; 1];
               end
           end       
           Xtrain = [Xtrain; i + 1*rand(1,1) j + 1*rand(1,1)];
       end
   end

  Xtest = [];
  ytest = [];
   for i = linspace(-1, 1, 15);
       for j = linspace(-1, 1, 15);
           if (i < 0);
               ytest = [ytest; -1];
           elseif (i > 0);
               ytest = [ytest; 1];
           else
               if (rand(1,1) <= 0.5);
                   ytest = [ytest; -1];
               else
                   ytest = [ytest; 1];
               end
           end
           Xtest = [Xtest; i j];
       end
   end
  