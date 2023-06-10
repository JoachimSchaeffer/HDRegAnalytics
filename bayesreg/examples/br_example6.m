%% Example 6
%  This example demonstrates how to use bayesreg to fit Bayesian logistic
%  regression models using data stored in a MATLAB table.
clear;

fprintf('Example 6 - Bayesian logistic regression with MATLAB tables\n');

%% Load the data 
n = 100;    % sample size
[~, Tb, X, ~, yb] = br_create_example_table(n);

%% Fit models
% Bayesian logistic regression using matrices; 'catvars' indicates which
% variables are categorical in the matrix X.
[b, b0, retval] = bayesreg(X,yb,'binomial','lasso','nsamples',1e4,'burnin',1e4,'thin',5,'catvars',[4,7],'display',true);

% Note: the warning here demonstrates what happens if you train on data
% with categorical variables in which not all categories are represented --
% Bayesreg gives a warning to let you know you may have problems predicting
% on new data if it contains the missing category

% Bayesian logistic regression using the same data stored in a MATLAB table (outcome variable is Diabetes)
[b, b0, retval] = bayesreg(Tb,'Diabetes','binomial','lasso','nsamples',1e4,'burnin',1e4,'thin',5,'display',true);

%% Compute predictions
fprintf('\n');
fprintf('----------------------------------------------------------------------\n');
[pred, predstats] = br_predict(Tb, b, b0, retval, 'CI', [2.5, 97.5], 'display', true);