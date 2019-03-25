%% Synthesize Data

nInstances = 500;
nVariables = 2;
nClasses = 5;

% Generate Parameters
wTrue = randn(nVariables,1);
gammaTrue = sort(randn(nClasses-1,1));

% Generate Data synthetic data ordered along 1 dimension
Xtrain = randn(nInstances,nVariables);
z = Xtrain*wTrue;
ytrain = zeros(nInstances,1);

% assign class labels
ytrain(z < gammaTrue(1)) = 1;
for class = 2:nClasses-1
    ytrain(z >= gammaTrue(class-1) & z < gammaTrue(class)) = class;
end
ytrain(z >= gammaTrue(nClasses-1)) = nClasses;

%% Preprocess Data

[n,d] = size(Xtrain);

% Standardize features and add bias
Xtrain = standardizeCols(Xtrain);
Xtrain = [ones(nInstances,1) Xtrain];

% Split into training/test set
perm = randperm(n);
Xtest = Xtrain(perm(ceil(n/2)+1:end),:);
ytest = ytrain(perm(ceil(n/2)+1:end));
Xtrain = Xtrain(perm(1:ceil(n/2)),:);
ytrain = ytrain(perm(1:ceil(n/2)),:);
n = size(Xtrain,1);
