%% Description
% Demonstrates parameter learning and inference on Hidden Markov Models
% using classic problem setup of a Casino that switches out a fair die for
% a loaded die

close all
clear all

% ----- Generate Sequence Data --------------------------------------------
% "Occassionally dishonest casino" example
O = [1/6 , 1/6 , 1/6 , 1/6 , 1/6 , 1/6  ;  ... % fair die
     1/10, 1/10, 1/10, 1/10, 1/10, 5/10 ];   % loaded die
T = [1/2 , 1/2; 1/4, 3/4];     
pi = [1/2, 1/2];
len = 25;
numSeq = 1;
seq = ml_sampleMC(T, pi, numSeq, len);
X = zeros(numSeq, len);
% generate observation states from distribution at each hidden state
for k = 1:len
    for p = 1:numSeq
        X(p, k) = ml_randSampleDiscrete(O(seq(k), :));
    end
end

%% Usage with known observation and transition probability matrix: Viterbi
options_hmm = [];
options_hmm.observed = O;
options_hmm.transition = T;
options_hmm.pi = pi;
model_hmm = ml_unsupervised_HMM(X, options_hmm);
viterbiSeq = model_hmm.Viterbi(model_hmm);
hmmviterbi(X, T, O)
individuallyBestSeq = model_hmm.individuallyMostLikely(model_hmm);
disp('Individually most likely latent variable states:')
disp(individuallyBestSeq);
disp('True latent variable states:')
disp(seq)
disp('Viterbi most likely latent variable sequence:')
disp(viterbiSeq)
disp('Misclassification rate for Viterbi:')
disp(sum(viterbiSeq ~= seq) / length(seq))
disp('Misclassification rate for Most Likely Individual States:')
disp(sum(individuallyBestSeq ~= seq) / length(seq))

%% Usage with unknown parameters: Baum Welch (EM) algorithm
% Learn the parameters of the occassionaly dishonest casino based only on
% observations and structure assumption
O = [1/6 , 1/6 , 1/6 , 1/6 , 1/6 , 1/6;  ...  % fair die
     1/10, 1/10, 1/10, 1/10, 1/10, 5/10 ];    % loaded die
T = [0.50 , 0.50; 0.25, 0.75];
len = 100; % if len is too short, may not observe something in the sequence
numSeq = 50;
seq = ml_sampleMC(T, pi, numSeq, len);
obs = zeros(numSeq, len);
[~, seqLen] = size(O);

% generate observation states from distribution at each hidden state
X = zeros(numSeq, len);
for k = 1:len
    for p = 1:numSeq
        X(p, k) = ml_randSampleDiscrete(O(seq(k), :));
    end
end

options_hmm_unknown = [];
options_hmm_unknown.nHiddenStatesGuess = 2;
options_hmm_unknown.nObservableStatesGuess = 6;
model_hmm_unknown = ml_unsupervised_HMM(X, options_hmm_unknown);
[Ohat, That] = model_hmm_unknown.BaumWelch(model_hmm_unknown);
disp('True emission probabilities:')
disp(O);
disp('Learned emissions probabilities:')
disp(Ohat);
disp('True transition probabilities:')
disp(T);
disp('Learned transition probabilities:')
disp(That);

