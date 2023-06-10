function lambda2 = sample_lambda2_hs(b, sigma2, tau2, nu, delta2prod)
%SAMPLE_LAMBDA2_HS samples the hyperparameters lambda2 for HS model.
%   lambda2 = sample_lambda2_hs(...) samples the lambda2 hyperparameters
%   from the conditional posterior distribution. Horseshoe prior for
%   regression coefficients beta.
%
%   The input arguments are:
%       b       - [p x 1] regression coefficients
%       sigma2  - [1 x 1] noise variance
%       tau2    - [1 x 1] global variance hyperparameter
%       nu      - [p x 1] hyperparameters
%
%   Return values:
%       lambda2 - a sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

scale = 1./nu + b.^2./2./tau2./sigma2./delta2prod;
lambda2 = 1 ./ exprnd_fast(1./scale);

end