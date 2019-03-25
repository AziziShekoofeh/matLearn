Xtrain1 = linspace(-5, -2, 30)';
ytrain1 = 0.05*randn(30,1)+Xtrain1./(1+abs(Xtrain1));
Xtest1 = linspace(-5, -2, 50)';
ytest1 = Xtest1./(1+abs(Xtest1));

Xtrain2 = linspace(-2, 2, 30)';
ytrain2 = 0.4*randn(30,1)+10*exp(-5*Xtrain2.^2);
Xtest2 = linspace(-2, 2, 50)';
ytest2 = 10*exp(-5*Xtest2.^2);

Xtrain3 = linspace(2, 5, 30)';
ytrain3 = 0.5*rand(30, 1) + Xtrain3.^2;

Xtest3 = linspace(2, 5, 50)';
ytest3 = Xtest3.^2;

Xtrain = [Xtrain1; Xtrain2; Xtrain3];
ytrain = [5*ytrain1; ytrain2; 0.5*ytrain3];
Xtest = [Xtest1; Xtest2; Xtest3];
ytest = [5*ytest1; ytest2; 0.5*ytest3];