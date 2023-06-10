function [sigma2, muSigma2, e, mu] = sample_sigma2(mu, y, b, ete, omega2, tau2, lambda2, delta2prod, gprior)
%SAMPLE_SIGMA2 samples the error variance for linear regression models.
%   [sigma2, muSigma2, e] = sample_sigma2(...) samples variance sigma2
%   from the conditional posterior distribution. Linear regression models.
%
%   The input arguments are:
%       X        - [n x p] data matrix
%       y        - [n x 1] target vector
%       b        - [p x 1] regression parameters
%       b0       - [1 x 1] intercept parameter
%       omega2   - [n x 1] error model hyperparameters
%       lambda2  - [p x 1] local variance hyperparameters
%       tau2     - [1 x 1] global variance hyperparameter
%       bXtXb    - [1 x 1] b'*(X'X)*b (used for gprior only)
%       gprior   - [1 x 1] true for gprior, otherwise false
%
%   Return values:
%       sigma2   - [1 x 1] sample from the posterior distribution
%       muSigma2 - [1 x 1] posterior mean
%       e        - [n x 1] error vector
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

n = length(y);
p = length(b);

%% Update of sigma2
e = [];
if (isempty(ete))
    % Compute errors, if required
    e = y - mu;
    ete = sum(e.^2./omega2);
end

shape = (n + p) / 2;

% g-prior has a different sampler for sigma2 compared to the other priors
if(~gprior)
    scale = ete/2 + sum(b.^2 ./ lambda2 ./ delta2prod)/2/tau2;
else
    bXtXb = mu'*mu;
    scale = sum(e.^2 ./ omega2)/2 +  bXtXb/tau2/2;
end
sigma2 = scale / randg(shape,1);

%% Posterior mean
muSigma2 = scale / (shape - 1);

end