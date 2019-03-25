nInstances = 600;
nVariables = 2;
nLabels = 5;

% Generate Parameters
wTrue = randn(nVariables,nLabels);

% Generate Data
X = randn(nInstances,nVariables);
y = binary2LinearInd(sign(X*wTrue));

%% Preprocess Data
[n,p] = size(X);

% Standardize features and add bias
X = standardizeCols(X);
X = [ones(nInstances,1) X];

% Split into training/test set
perm = randperm(n);
Xtest = X(perm(ceil(n/2)+1:end),:);
ytest = y(perm(ceil(n/2)+1:end),:);
Xtrain = X(perm(1:ceil(n/2)),:);
ytrain = y(perm(1:ceil(n/2)),:);
n = size(X,1);