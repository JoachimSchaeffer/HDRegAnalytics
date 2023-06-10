%% Example 14
% Another example showing how blocksampling can be used to speed up
% sampling if p is large
clear;

fprintf('Example 14 - p=500, n=5000 and block sampling of beta\n');

%% Generate data
n = 5e3;
p = 5e2;
X = randn(n,p);
beta = [zeros(p-5,1); ones(5,1)];
y = X*beta + randn(n,1);

%% Sample betas without using block-sampling
fprintf('Generating 1000 samples for p=500 betas and n=5000 datapoints, without block sampling.\n');
tic;
bayesreg(X,y,'gaussian','hs','nsamples',1e3,'burnin',5,'thin',1,'display',false);
toc;


%% Sample betas using 30 blocks
fprintf('\n');
fprintf('Generating 1000 samples for p=500 betas and n=5000 datapoints, using 30 blocks.\n');

tic;
bayesreg(X,y,'gaussian','hs','nsamples',1e3,'burnin',5,'thin',1,'display',false,'blocksample',30);
toc;