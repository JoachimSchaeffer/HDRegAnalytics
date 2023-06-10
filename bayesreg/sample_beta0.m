function [b0, m] = sample_beta0(X, z, mu_z, Xt1, beta, sigma2, omega2)
%SAMPLE_BETA0 samples the intercept parameter beta0.
%   [b0, m] = sample_beta0(...) samples the intercept parameter
%   beta0 from the conditional posterior distribution.
%
%
%   The input arguments are:
%       X       - [n x p] data matrix 
%       z       - [n x 1] target vector
%       beta    - [p x 1] regression parameters
%       mu_z    - [1 x 1] precomputed mean of z (empty if not used)
%       Xt1     - [p x 1] precomputed X'*1n (empty if not used, only used if no normalization of X)
%       sigma2  - [1 x 1] noise variance
%       omega2  - [n x 1] hyperparameters
%
%   Return values:
%       b0      - a beta0 sample from the posterior distribution
%       m       - posterior mean
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%% If precomputed mean(z)
if (~isempty(mu_z))
    n = length(z);
    if (isempty(Xt1))
        m = mu_z;
    else
        m = mu_z - beta'*Xt1/n;
    end
    v = sigma2/n;
else
    W = sum(1./omega2);
    m = sum((z - X*beta) ./ omega2) / W;
    v = sigma2 / W;
end

b0 = m + randn(1)*sqrt(v);

end