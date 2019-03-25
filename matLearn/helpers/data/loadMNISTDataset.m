% loadMNISTDataset

% Requires files to be compressed in gunzip format
trainImFile='train-images-idx3-ubyte';
trainLabFile='train-labels-idx1-ubyte';
testImFile='t10k-images-idx3-ubyte';
testLabFile='t10k-labels-idx1-ubyte';
gunzip(strcat([trainImFile,'.gz']));
gunzip(strcat([trainLabFile,'.gz']));
gunzip(strcat([testImFile,'.gz']));
gunzip(strcat([testLabFile,'.gz']));

fprintf('Decompressing MNIST files...');
pause(1) % give OS time to update
fprintf('Done.\nLoading MNIST files into Workspace...');
Xtrain = ml_readMNIST(trainImFile, 'images');
ytrain = ml_readMNIST(trainLabFile, 'labels');
Xtest = ml_readMNIST(testImFile, 'images');
ytest = ml_readMNIST(testLabFile, 'labels');
fprintf('Ready to train model\n');

MAXPIXEL = 255;
nClasses = 10;
Xtrain = double(Xtrain)./MAXPIXEL;
Xtest = double(Xtest)./MAXPIXEL;
ytrain(ytrain == 0) = nClasses; % Matlab is 1-indexed, modify class 0 to 10
ytest(ytest == 0) = nClasses;

% TODO: delete uncompressed files after loading 