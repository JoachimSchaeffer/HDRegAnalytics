function [tau2, muTau2] = sample_tau2_trunc(b, sigma2, lambda2, delta2prod, mu, gprior, n)
%SAMPLE_TAU2 samples the global variance hyperparameter for all models.
%   [tau2, muTau2] = sample_tau2(...) samples global variance tau2
%   from the conditional posterior distribution. All models.
%
%   The input arguments are:
%       b        - [p x 1] regression parameters
%       sigma2   - [1 x 1] variance parameter
%       lambda2  - [p x 1] local variance hyperparameters
%       mu       - [1 x 1] b'*(X'X)*b (used for gprior only)
%       gprior   - [1 x 1] true for gprior, otherwise false
%
%   Return values:
%       tau2   - [1 x 1] sample from the posterior distribution
%       muTau2 - [1 x 1] posterior mean
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

p = length(b);

%% Update of tau2
if(~gprior)
    m = sum((b.^2 ./ lambda2 ./ delta2prod)/2/sigma2);
else
    m = mu'*mu/2/sigma2;    
end
tau2 = rejsample_hs_p_ab_trunc(m, p, 1/2, 1/2, log(1/p)/2, log(n)/2);

muTau2 = tau2;