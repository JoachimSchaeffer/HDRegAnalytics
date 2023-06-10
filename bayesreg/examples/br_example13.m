%% Example 13
%  This example demonstrates how to use the blocksampling option to handel
%  data sets with very large p. We also show how to specify the prior
%  distribution for the tau2 parameter.
clear;

fprintf('Example 13 - Large p and block sampling of beta\n');

%% Generate data
n = 1e2;
p = 5e4;
X = randn(n,p);
beta = [zeros(p-5,1); ones(5,1)];
y = X*beta + randn(n,1);

%% Sample betas using 5 blocks
fprintf('Sample p=50,000 betas using 5 blocks.\n');
tic;
[b, b0, retval] = bayesreg(X,y,'gaussian','hs','nsamples',5,'burnin',5,'thin',1,'display',false,'blocksample',5);
toc;


%% Sample betas using block of size ~10,000 and the Strawderman-Berger prior for tau2
fprintf('\n');
fprintf('Sample p=50,000 betas using blocks of size 10,000. Use the ''Strawderman-Berger'' prior for tau2.\n');

tic;
bayesreg(X,y,'gaussian','hs','nsamples',5,'burnin',5,'thin',1,'display',false,'blocksize',1e4,'tau2prior','sb');
toc;