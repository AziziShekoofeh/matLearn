function [s] = ml_randSampleDiscrete(pi)
% Requires pi to be normalized discrete probability distribution
s = find(cumsum(pi) > rand, 1);
end
