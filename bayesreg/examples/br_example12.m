%% Example 12
% This example demonstrates how to use the "groups" option in bayesreg. 
% This example shows that you can have one variable in multiple groups
% which may be useful for genomics applications (e.g., grouping a genomic
% variant into potentially overlapping genes) or natural language
% processing in which a word may belong to multiple categories (e.g., noun,
% words about France, words about travel, etc.)
%
%
clear;

fprintf('Example 12 - Bayesian linear regression with groups\n');

rng(1);

%% Create a design matrix
n = 100;
p = 5;
X = randn(100, 5);
beta = ones(p, 1);
y = X*beta + randn(n,1);

fprintf('\n');
fprintf('We have 5 predictors grouped into 4 groups.\n');
fprintf('Predictor 1  2  3  4  5\n');
fprintf(' Group 1  *  *\n');
fprintf(' Group 2  *     *\n');
fprintf(' Group 3  *  *  *\n');
fprintf(' Group 4        *  *  *\n\n');
fprintf('To do this we set the groups option to {[1 2], [1 3], [1 2 3], [3 4 5]}\n\n');

%% Fit data using bayesreg with knowledge of groups and compute predictions
fprintf('\n');
fprintf('Running bayesreg with the groups options ... \n');
groups = {[1 2], [1 3], [1 2 3], [3 4 5]};
[b, b0, retval] = bayesreg(X,y,'gaussian','hs','nsamples',1e4,'burnin',1e4,'thin',5,'display',true,'groups',groups);

%% bayesreg group structure
retval.grouping


