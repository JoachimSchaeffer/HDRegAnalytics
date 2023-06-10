function [x, m, L] = fastmvg_rue(Phi, PtP, alpha, Ptalpha, D, sigma2, omega, gprior, XtX)
%FASTMVG_RUE sampler for multivariate Gaussians. 
%   [x, m, L] = fastmvg_rue(...) generates multivariate Gaussian random
%   variates of the form: N(mu, S), where
%       mu = S Phi' y
%       S  = inv(Phi'Phi + inv(D))
%
%   Here, PtP = Phi'*Phi (X'X may be precomputed). Sampler used for n
%   larger than p.
%   
%   The input arguments are:
%       Phi     - [n x p] matrix 
%       PtP     - [p x p] matrix (precomputed if available, [] otherwise)
%       alpha   - [p x 1] vector
%       Ptalpha - [p x 1] vector (precomputed if available, [] otherwise)
%       D       - [p x 1] vector
%       sigma2  - [1 x 1] vector
%       omega   - [n x 1] vector ([] if not needed)
%       gprior  - [1 x 1] true for gprior, otherwise false
%
%   Return values:
%       x      - [p x 1] vector of MVG random variates
%       m      - [p x 1] vector, posterior mean of the MVG 
%       L      - [p x p] Cholesky factor 
%
%   Reference:
%     Rue, H. (2001). Fast sampling of gaussian markov random fields. Journal of the Royal
%     Statistical Society: Series B (Statistical Methodology) 63, 325–338.
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%% Compute matrices, if not passed
if(isempty(PtP))
    PtP = Phi' * Phi;
end
if (isempty(Ptalpha))
    Ptalpha = Phi' * (alpha./omega);
end

p = length(D);

if(~gprior)
    L = chol(PtP/sigma2 + diag(1./D), 'lower');
else
    % If XtX precomputed and passed
    L = chol(PtP/sigma2 + XtX./D(1), 'lower');
end
v = L \ (Ptalpha/sigma2);
m = L' \ v;
w = L' \ randn(p,1);

x = m + w;

end