function w2 = sample_w2_logt(loglambda, psi2, alpha)
%SAMPLE_W2_LOGT samples the hyperparameters w2 for the log-t logscale prior.
%   w2 = sample_w2_logt(...) samples the w2 hyperparameters
%   from the conditional posterior distribution for the log-t prior.
%
%   The input arguments are:
%       loglambda - [p x 1] logarithm of lambda hyperparameters
%       psi2      - [1 x 1] logscale prior scale parameter
%       alpha     - [1 x 1] log-t degrees of freedom
%
%   Return values:
%       w2        - [p x 1] sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2018

p = length(loglambda);

scale = loglambda.^2/2/psi2 + alpha/2;
shape = (alpha + 1)/2*ones(p,1);
w2 = scale ./ randg(shape);

end