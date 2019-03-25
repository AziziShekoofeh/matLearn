function [samples] = ml_sampleMC(A, pi, nSamples, lenSequence)
samples = zeros(nSamples, lenSequence);
for i = 1:nSamples
    samples(i, 1) = ml_randSampleDiscrete(pi); % starting state
    for j = 2:lenSequence
        % randomly jump to next state from currentState
        samples(i, j) = ml_randSampleDiscrete(A(samples(i,j-1), :));
    end
end
end
