%% Synthesize Data

nInstances = 500;
nVariables = 2;
nClasses = 5;

% Generate Parameters
Xtrain = randn(nInstances,nVariables);
nExamplePoints = 3;
examplePoints = randn(nExamplePoints,nVariables);
% Generate random class thresholds
thresholds = [0;cumsum(2*rand(nClasses-1,1))];
% Assign class labels by thresholding the smallest L2 distance to example
% points for each row x_i
for i = 1:nInstances
    dists = sum((repmat(Xtrain(i,:),nExamplePoints,1) - examplePoints).^2,2);
    ytrain(i,1) = max(find(min(dists) > thresholds));
end

%% Preprocess Data

[n,p] = size(Xtrain);

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
