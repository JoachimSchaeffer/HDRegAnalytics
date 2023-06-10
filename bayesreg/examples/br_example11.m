%% Example 11
% This example demonstrates how to use the "groups" option in bayesreg.
% In this example we expand columns of the design matrix using 
% Legendre polynomials. The new columns are then grouped using
% the "groups" option. We then compare the prediction performance of the
% grouped vs the non-grouped model.
% Grouping of variables works only with HS, HS+ and lasso priors. Note that
% the same variable can appear in multiple groups.
%
%   References:
%     Xu, Z., Schmidt, D.F., Makalic, E., Qian, G. & Hopper, J.L.
%     AI 2016: Advances in Artificial Intelligence, pp. 229-240, 2016
%
clear;

fprintf('Example 11 - Bayesian linear regression with groups\n');

rng(1);

%% Create a design matrix
n = 50;
p = 4;
X = rand(n,p) * 2 - 1;          % each x is in [-1, 1]
Xt = rand(1e4,p) * 2 - 1;    

%% Expand this design matrix using Legendre polynomials
df = 3;
Z = [];
Zt = [];
for i = 1:p
    Z = [Z polyexpand(X(:,i), df)];
    Zt = [Zt polyexpand(Xt(:,i), df)];
end

% Create some data y 
% Note that polynomials (groups) 1 and 3 are associated with y, whereas 
% polynomials 2 and 4 are not associated with y.
beta = [ ones(df,1); zeros(df,1); ones(df,1); zeros(df,1) ];
snr = 2;
s2 = var(Z*beta) / snr;
y = Z*beta + randn(n,1)*sqrt(s2);
yt = Zt*beta + randn(1e4,1)*sqrt(s2);

%% Fit data using bayesreg with knowledge of groups and compute predictions
fprintf('\n');
fprintf('Running bayesreg with the groups options ... \n');
[b, b0, retval] = bayesreg(Z,y,'gaussian','hs','nsamples',1e4,'burnin',1e4,'thin',5,'display',true,'groups',{1:3,4:6,7:9,10:12});
fprintf('\n');
[p1, pstat1] = br_predict(Zt, b, b0, retval, 'CI', [2.5, 97.5], 'ytest', yt, 'display', true);
fprintf('\n');

%% Fit data using bayesreg with no grouping and compute predictions
fprintf('\n');
fprintf('Running bayesreg with no groups ... \n');
[b, b0, retval] = bayesreg(Z,y,'gaussian','hs','nsamples',1e4,'burnin',1e4,'thin',5,'display',true);
fprintf('\n');
[p2, pstat2] = br_predict(Zt, b, b0, retval, 'CI', [2.5, 97.5], 'ytest', yt, 'display', true);
fprintf('\n');

%% Compare negative log-likelihoods on new data of the two models
fprintf('\n');
fprintf('----------------------------------------------------------------------\n');
fprintf('Difference in negative log-likelihood (groups vs nogroups) = %.3f\n', (pstat1.neglike - pstat2.neglike));
