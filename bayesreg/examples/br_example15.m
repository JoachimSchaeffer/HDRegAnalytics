%% Example 15
%  This example demonstrates how to use br_sparsify() to produce sparse
%  estimates of the regression coefficients following MCMC sampling with
%  bayesreg.
clear;

fprintf('Example 15 - Demonstration of how to sparsify Bayesian linear regression estimates\n');

%% Load the data 
n = 100;    % sample size
[T, ~, X, y] = br_create_example_table(n);

%% Fit models
% Bayesian linear regression using a MATLAB table (outcome variable is Diabetes)
[b, b0, retval] = bayesreg(T,'Diabetes','gaussian','lasso','nsamples',1e4,'burnin',1e4,'thin',5,'display',true);

%% Compute sparse estimates
fprintf('\n');
fprintf('Sparsification of estimates with the SAVS algorithm\n');
fprintf('----------------------------------------------------------\n');
br_sparsify(T, y, b, b0, retval,'method','savs','display',true);

fprintf('\n');
fprintf('Sparsification based on k-means clustering\n');
fprintf('----------------------------------------------------------\n');
retval = br_sparsify(T, y, b, b0, retval,'method','kmeans','display',true);

fprintf('\n');
fprintf('Sparsification based on credible intervals\n');
fprintf('----------------------------------------------------------\n');
retval = br_sparsify(T, y, b, b0, retval,'method','ci','display',true);


%% Compute predictions
fprintf('\n');
fprintf('Compute predictions with sparse estimates (obtained from the CI method)\n');
fprintf('----------------------------------------------------------------------\n');
[pred, predstats] = br_predict(T, b, b0, retval, 'display', true, 'predictor', 'sparse');