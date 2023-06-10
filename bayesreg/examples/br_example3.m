%% Example 3
%  This example demonstrates how to use bayesreg to fit Bayesian logistic
%  regression models with the horseshoe prior.
clear;

fprintf('Example 3 - Bayesian logistic regression using the Pima Indians data\n');

%% Load data
load pima.mat

%% Bayesian logistic regression (horseshoe prior)
[beta, beta0, retval] = bayesreg(X,y,'binomial','hs','nsamples',1e4,'burnin',1e4,'thin',5,'displayor',true,'varnames',varnames,'display',false);

%% Display sampling statistics
br_summary(beta,beta0,retval);

%% Get statistics on how well we fit the data
fprintf('\n');
fprintf('----------------------------------------------------------------------\n');
[pred_pima, predstats_pima] = br_predict(X, beta, beta0, retval, 'ytest', y, 'display', true);
