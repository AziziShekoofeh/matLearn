  Xtrain = [];
  ytrain = [];
   for i = linspace(-1, 1, 12);
       for j = linspace(-1, 1, 12);
           if (i-j < 0);
               ytrain = [ytrain; -1];
           elseif (i-j > 0);
               ytrain = [ytrain; 1];
           else
               if (randn(1,1) < 0);
                   ytrain = [ytrain; -1];
               else
                   ytrain = [ytrain; 1];
               end
           end
                   
           Xtrain = [Xtrain; i + (2*rand(1,1)-1) j + (2*rand(1,1)-1)];
       end
   end

  Xtest = [];
  ytest = [];
   for i = linspace(-1, 1, 15);
       for j = linspace(-1, 1, 15);
           if (i-j < 0);
               ytest = [ytest; -1];
           elseif (1-j > 0);
               ytest = [ytest; 1];
           else
               if (randn(1,1) < 0);
                   ytest = [ytest; -1];
               else
                   ytest = [ytest; 1];
               end
           end
           Xtest = [Xtest; i j];
       end
   end
  
