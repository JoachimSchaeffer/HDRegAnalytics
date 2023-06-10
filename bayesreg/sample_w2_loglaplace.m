function w2 = sample_w2_loglaplace(loglambda, psi2)
%SAMPLE_W2_LOGT samples the hyperparameters w2 for the log-Laplace logscale prior.
%   w2 = sample_w2_logt(...) samples the w2 hyperparameters
%   from the conditional posterior distribution for the log-Laplace prior.
%
%   The input arguments are:
%       loglambda - [p x 1] logarithm of lambda hyperparameters
%       psi2      - [1 x 1] logscale prior scale parameter
%
%   Return values:
%       w2        - [p x 1] sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2019

mu = sqrt(2*psi2./loglambda.^2);
lambda = 1/2;
w2 = 1./randinvg(mu, lambda);

end