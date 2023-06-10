function lambda2 = sample_lambda2_lasso(b, sigma2, tau2, delta2prod)
%SAMPLE_LAMBDA2_LASSO samples the hyperparameters lambda2 for lasso model.
%   lambda2 = sample_lambda2_lasso(...) samples the lambda2 hyperparameters
%   from the conditional posterior distribution. Lasso prior for
%   regression coefficients beta.
%
%   The input arguments are:
%       b       - [p x 1] regression coefficients
%       sigma2  - [1 x 1] noise variance
%       tau2    - [1 x 1] global variance hyperparameter
%
%   Return values:
%       lambda2 - [p x 1] sample from the posterior distribution
%
%   References:
%     Park, T. & Casella, G. 
%     The Bayesian Lasso 
%     Journal of the American Statistical Association, Vol. 103, pp. 681-686, 2008
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

mu = sqrt(2 * tau2 * sigma2 .* delta2prod ./ b.^2);
shape = 2;
lambda2 = 1 ./ randinvg(mu, 1/shape);

end