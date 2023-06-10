%% Example 8
%  This example demonstrates how to use bayesreg to fit Bayesian linear
%  regression models using data stored in a MATLAB table.
clear;

fprintf('Example 8 - Bayesian linear regression with MATLAB tables\n');

rng(1);

%% Load the data 
n = 100;    % sample size
[T, ~, X, y] = br_create_example_table_ex8(n);

%% Fit models
% Bayesian linear regression using the same data stored in a MATLAB table (outcome variable is Diabetes)
[b, b0, retval] = bayesreg(T,'Diabetes','gaussian','hs','nsamples',1e4,'burnin',1e4,'thin',5,'display',true,'nogrouping',false);

%% Compute predictions
fprintf('\n');
fprintf('----------------------------------------------------------------------\n');
[pred, predstats] = br_predict(T, b, b0, retval, 'CI', [2.5, 97.5], 'display', true);
