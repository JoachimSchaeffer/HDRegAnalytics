function [tau2, muTau2] = sample_tau2(b, sigma2, lambda2, delta2prod, xi, mu, gprior, tau_a)
%SAMPLE_TAU2 samples the global variance hyperparameter for all models.
%   [tau2, muTau2] = sample_tau2(...) samples global variance tau2
%   from the conditional posterior distribution. All models.
%
%   The input arguments are:
%       b        - [p x 1] regression parameters
%       sigma2   - [1 x 1] variance parameter
%       lambda2  - [p x 1] local variance hyperparameters
%       xi       - [1 x 1] global hyperparameter
%       bXtXb    - [1 x 1] b'*(X'X)*b (used for gprior only)
%       gprior   - [1 x 1] true for gprior, otherwise false
%
%   Return values:
%       tau2   - [1 x 1] sample from the posterior distribution
%       muTau2 - [1 x 1] posterior mean
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

p = length(b);

%% Update of tau2
shape = p/2 + tau_a;
if(~gprior)
    scale = 1/xi + sum(b.^2 ./ lambda2 ./ delta2prod)/2/sigma2;
else
    scale = 1/xi + mu'*mu/2/sigma2;
end
tau2 = scale / randg(shape,1);

%% Posterior mean
muTau2 = scale / (shape - 1);

end