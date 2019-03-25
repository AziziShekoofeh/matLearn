function initializeMatLearn()
%% Run this script to initialize matLearn

isMatlab = ~isempty(ver('matlab'));
if ~isMatlab
    fprintf(1, ['Warning: matLearn doesn''t support Octave yet! Some ', ...
                'functions may not work.\n']);
end
disp('Initializing MatLearn...');

%% change Matlab's working directory to this script's directory
w = which(mfilename()); 
thisDir = fileparts(w);
cd(thisDir);
addpath(thisDir);

%% include all MatLearn files in path
addpath(genpath(pwd()))
disp('MatLearn is ready to use!');

end
