function [lambda2, loglambda] = sample_lambda2_logscale(b, sigma2, tau2, delta2prod, w2, psi2)
%SAMPLE_LAMBDA2_LOGSCALE samples the hyperparameters lambda2 for a log-scale model using a rejection sampler
%   lambda2 = sample_lambda2_logscale(...) samples the lambda2 hyperparameters
%   from the conditional posterior distribution using rejection sampler.
%   Log-scale prior for regression coefficients beta.
%
%   The input arguments are:
%       b              - [p x 1] regression coefficients
%       sigma2         - [1 x 1] noise variance
%       tau2           - [1 x 1] global variance hyperparameter
%       delta2prod     - [p x 1] vector of group shrinkage parameters
%                                associated with each coefficient
%       w2             - [p x 1] vector of log-scale variances
%       psi2           - [1 x 1] log-scale prior overall scale parameter
%
%   Return values:
%       lambda2 - a sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2018

m = b.^2./2./tau2./sigma2./delta2prod;
loglambda = rejsample_logscale(m, w2*psi2);
lambda2 = exp(2*loglambda);
       
end