function omega2 = sample_omega2_logistic(mu)
%SAMPLE_OMEGA2_LOGISTIC samples the hyperparameters omega2 for logistic regression.
%   omega2 = sample_omega2_logistic(...) samples the omega2 hyperparameters
%   from the conditional posterior distribution. Logistic regression.
%
%   This package uses the PolyaGamma sampler written by Jesse Windle - jwindle@ices.utexas.edu
%
%   The input arguments are:
%       mu      - [n x 1] linear predictor (log-odds)
%
%   Return values:
%       omega2  - [n x 1] sample from the posterior distribution
%
%   References:
%     Polson, N. G.; Scott, J. G. & Windle, J. 
%     Bayesian inference for logistic models using Pólya-Gamma latent variables 
%     Journal of the American Statistical Association, Vol. 108, 1339-1349, 2013
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

% PG(1, X beta + beta0)
omega2 = 1 ./ pgdraw(mu);


end