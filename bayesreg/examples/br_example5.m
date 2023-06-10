%% Example 5
%  This example demonstrates how to use bayesreg to fit Bayesian linear
%  regression models using data stored in a MATLAB table.
clear;

fprintf('Example 5 - Bayesian linear regression with MATLAB tables\n');

%% Load the data 
n = 50;    % sample size
[T, ~, X, y] = br_create_example_table(n);

%% Fit models
% Bayesian linear regression using matrices; 'catvars' indicates which
% variables are categorical in the matrix X.
[b, b0, retval] = bayesreg(X,y,'gaussian','lasso','nsamples',1e4,'burnin',1e4,'thin',5,'catvars',[4,7],'display',true);

% Bayesian linear regression using the same data stored in a MATLAB table (outcome variable is Diabetes)
[b, b0, retval] = bayesreg(T,'Diabetes','gaussian','lasso','nsamples',1e4,'burnin',1e4,'thin',5,'display',true);

%% Compute predictions
fprintf('\n');
fprintf('----------------------------------------------------------------------\n');
[pred, predstats] = br_predict(T, b, b0, retval, 'CI', [2.5, 97.5], 'display', true);