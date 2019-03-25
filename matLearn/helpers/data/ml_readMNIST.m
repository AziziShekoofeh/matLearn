function [Z] = ml_readMNIST(name, what)

% returns matrix of MNIST images or vector of labels
if strcmp(what, 'images')
    MNISTId = 2051; % first 4 bytes identify MNIST image dataset
elseif strcmp(what, 'labels')
    MNISTId = 2049;
end


mnistFile = fopen(name, 'rb');
if mnistFile == -1
    error(['Could not find file ', name]);
end

%  4-byte magic number identifying the file is in big-endian 'ieee-be'
%  http://yann.lecun.com/exdb/mnist/
magicNumber = fread(mnistFile, 1, 'int32', 0, 'ieee-be');
if magicNumber ~= MNISTId
    fclose(mnistFile);
    error(['Incorrect or damaged file to read ', what, ': ', name]);
end

if strcmp(what,'images')
    % next 12 bytes give dataset parameters in big-endian
    nImage = fread(mnistFile, 1, 'int32', 0, 'ieee-be');
    imRow = fread(mnistFile, 1, 'int32', 0, 'ieee-be');
    imCol = fread(mnistFile, 1, 'int32', 0, 'ieee-be');
    
    % read to end to get images
    raw = fread(mnistFile);
    % generate tensor of images in standard orientation
    imTensor = permute(reshape(raw, imCol, imRow, nImage),[2 1 3]);
    % reshape as vectors of size 1 x nFeature (== 1 x imCol*imRow)
    Z = reshape(imTensor, imRow*imCol, nImage)'; % nTrain x nFeature
    
elseif strcmp(what,'labels')
    fread(mnistFile, 1, 'int32', 0, 'ieee-be'); % ignore next number
    Z = fread(mnistFile);
end

fclose(mnistFile);
end
