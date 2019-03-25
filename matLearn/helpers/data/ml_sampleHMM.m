function[S] = ml_sampleHMM(O, T, p, nSample, seqLen)
hiddenStates = sampleMC(T, p, nSample, seqLen);
S = zeros(nSample, seqLen);
for n = 1:nSample
    for k = 1:seqLen
        S(n, k) = randSampleDiscrete(O(hiddenStates(k), :));
    end
end
end