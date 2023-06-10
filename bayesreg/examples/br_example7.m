%% Example 7
%  This example demonstrates how to use bayesreg to fit Bayesian logistic
%  regression models with categorical predictors, and predict onto new
%  (unseen) data
clear;

fprintf('Example 7 - Bayesian logistic regression using the Heart dataset, with categorical predictors and testing on new (unseen) data\n');

%% Load the data 
load heart;

%% Partition the data into training and testing
rng(1); c=cvpartition(size(X,1), 'holdout', 0.3);

%% Fit models
% Bayesian logistic regression using matrices without utilising the fact
% certain variables are categories
[b, b0, retval] = bayesreg(X(c.training,:),y(c.training,:),'binomial','lasso','nsamples',1e4,'burnin',1e4,'thin',5,'display',true,'varnames',varnames);

fprintf('\n----------------------------------------------------------------------\n');
fprintf('Prediction statistics without handling categorical predictors appropriately\n');
[pred, predstats] = br_predict(X(c.test,:), b, b0, retval, 'display', true, 'predict', 'bayesavg', 'ytest', y(c.test));
fprintf('\n');

% Bayesian logistic regression treating appropriate input variables as categorical predictors
[b, b0, retval] = bayesreg(X(c.training,:),y(c.training,:),'binomial','lasso','nsamples',1e4,'burnin',1e4,'thin',5,'display',true,'varnames',varnames,'catvars',[2,3,6,7,9,11,13]);

fprintf('\n----------------------------------------------------------------------\n');
fprintf('Prediction statistics handling categorical predictors appropriately\n');
[pred, predstats] = br_predict(X(c.test,:), b, b0, retval, 'display', true, 'predict', 'bayesavg', 'ytest', y(c.test));
fprintf('\n');
