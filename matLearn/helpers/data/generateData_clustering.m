% Generate clustering data
nInstances = 500;
nTest = 250;
nFeatures = 2;
nClusters = 5;
for c = 1:nClusters
    mu(:,c) = randn(nFeatures,1)*10;
    sigma(:,:,c) = randn(nFeatures);
    sigma(:,:,c) = sigma(:,:,c)+sigma(:,:,c)';
    sigma(:,:,c) = sigma(:,:,c)-(-1+min(min(eig(sigma(:,:,c)))))*eye(nFeatures);
    resp(c) = rand;
end
resp = resp/sum(resp);
Xtrain = zeros(nInstances,nFeatures);
Xtest = zeros(nTest,nFeatures);
for i = 1:nInstances
    c = sampleDiscrete(resp);
    Xtrain(i,:) = mvnrnd(mu(:,c),sigma(:,:,c),1);
    ytrain(i) = c;
end
for i = 1:nTest
    c = sampleDiscrete(resp);
    Xtest(i,:) = mvnrnd(mu(:,c),sigma(:,:,c),1);
end