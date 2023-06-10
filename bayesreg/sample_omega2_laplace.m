function omega2 = sample_omega2_laplace(e, sigma2)
%SAMPLE_OMEGA2_LAPLACE samples the hyperparameters omega2 for the Laplace error model.
%   omega2 = sample_omega2_laplace(...) samples the omega2 hyperparameters
%   from the conditional posterior distribution. Laplace error model.
%
%   The input arguments are:
%       e       - [n x 1] errors
%       sigma2  - [1 x 1] residual variance
%
%   Return values:
%       nu      - [n x 1] sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

mu = sqrt(2 * sigma2 ./ e.^2);
lambda = 2;

omega2 = 1 ./ randinvg(mu, 1/lambda);

end