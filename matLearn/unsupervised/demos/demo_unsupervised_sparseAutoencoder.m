loadMNISTDataset

imsize = sqrt(size(Xtrain,2));
Xtrain = Xtrain(1:10000,:);
inds = randi(size(Xtrain,1),200,1);
plotSquareImages(Xtrain(inds,:));

%% Usage of sparse Autoencoder model
options_sae = [];
options_sae.nHidden = [196];
% which hidden layers to add sparsity penalty
options_sae.sparsify = [1];
% sparsity penalty
options_sae.betas = [3];
% target average activations of hidden units in sparse layers expressed
% as an average activation
options_sae.rhos = [0.1];
% weight decay
options_sae.lambda = 3e-3;
model_sae = ml_unsupervised_sparseAutoencoder(Xtrain, options_sae);

%% Reconstruct original data using model
Xrecon = model_sae.predict(model_sae, Xtrain);
plotSquareImages(Xrecon(inds,:))
hiddenSize = model_sae.nHidden(1);
visibleSize = imsize.^2;

%% Visualize weights of final layer
W1 = reshape(model_sae.w(1:hiddenSize*visibleSize), visibleSize, hiddenSize);
plotSquareImages(W1');
