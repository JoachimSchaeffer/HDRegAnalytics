%% Example 4
%  This example demonstrates how to use bayesreg to fit Bayesian logistic
%  regression models with the horseshoe+ prior.
clear;

fprintf('Example 4 - Bayesian logistic regression using the Pima Indians data with interactions\n');

%% Load data
load pima.mat
Xs = x2fx(X, 'quadratic');
Xs(:,1) = [];

% Variable names
N = size(Xs, 2);
p = length(varnames);
vn = cell(N,1);

vn(1:p) = varnames;
it = p+1;
for i = 1:p-1
    for j=i+1:p
        vn{it} = [varnames{i}, 'x', varnames{j}];
        it = it + 1;
    end
end

for i = 1:p
   vn{it} = [varnames{i}, '^2'];
   it = it + 1;
end

%% Bayesian logistic regression (horseshoe prior)
[beta, beta0, retval] = bayesreg(Xs,y,'binomial','hs+','nsamples',1e4,'burnin',1e4,'thin',5,'varnames',vn,'sortrank',true);

%% Get statistics on how well we fit the data
[pred_pima_ix, predstats_pima_ix] = br_predict(Xs, beta, beta0, retval, 'ytest', y, 'display', true);
