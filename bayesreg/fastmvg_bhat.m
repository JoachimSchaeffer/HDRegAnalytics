 function [x,u] = fastmvg_bhat(Phi, alpha, D, sigma, omega)
%FASTMVG_BHAT sampler for multivariate Gaussians. 
%   [x, m, L] = fastmvg_bhat(...) generates multivariate Gaussian random
%   variates of the form: N(mu, S), where
%       mu = S Phi' y
%       S  = inv(Phi'Phi + inv(D))
%
%   Sampler used for p larger than n.
%   
%   The input arguments are:
%       Phi   - [n x p] matrix 
%       alpha - [p x 1] vector
%       D     - [p x 1] vector
%
%   Return values:
%       x     - [p x 1] vector of MVG random variates
%       u     - [p x 1] vector, posterior mean of the MVG  
%
%   Reference:
%
%   Fast sampling with Gaussian scale-mixture priors in high-dimensional regression
%   A. Bhattacharya, A. Chakraborty and B. K. Mallick
%   arXiv:1506.04778
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

[n,p] = size(Phi);

u = randn(p,1) .* sqrt(D);
delta = randn(n,1);
v = Phi*u/sigma + delta;
Dpt = bsxfun(@times, Phi', D)/sigma;
W = Phi*Dpt/sigma + eye(n);

w = W \ (alpha./omega/sigma - v);
x = u + Dpt*w;

%% HACK!
u = x;

end