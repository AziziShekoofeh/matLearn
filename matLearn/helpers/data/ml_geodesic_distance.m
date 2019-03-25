function [D] = ml_geodesic_distance(D, options) 
% Calculates geodesic (shortest-path) distance given a set of points X
% using the K nearest neighbours 
%
% Author: Geoffrey Roeder 

% Compute all distances


[K, disconnected] = myProcessOptions(options, 'K', 2, 'disconnected', 0);
[nInstances, nVars] = size(D);

%D = X.^2*ones(nVars,nInstances) + ones(nInstances,nVars)*(X').^2 - 2*(X*X');

% Find K nearest neighbours
% -------------------------------------------------------------------------
kNearest = zeros(nInstances,K);

for j=1:nInstances
    % b holds the indices of the sorted array
    [~,idx] = sort(D(:,j));
    
    % keep indices of the k smallest distances
    kNearest(j,:) = idx(2:K+1);
end

% Form sparse adjacency matrix G
G = sparse(nInstances,nInstances);
A = sparse(nInstances,nInstances);
for i = 1:nInstances
    % form the undirected graph
    G(i, kNearest(i,:)) = D(i,kNearest(i,:));
    A(i, kNearest(i,:)) = 1;
end
G = G;
A = A;

% Do Dijkstra's shortest path on sparse weighted graph of distances
% -------------------------------------------------------------------------
% NOTE: the distances matrix is upper triangular
D = zeros(nInstances);

[D, ~] = Dijkstra(A,G);


% set infinite distances returned by Dijkstra to max in graph
% -------------------------------------------------------------------------
if disconnected
    D(~isfinite(D)) = NaN;
    D(isnan(D)) = max(max(D));
end

end