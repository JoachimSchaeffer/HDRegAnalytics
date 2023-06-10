function out = randinvg(mu, lambda)
%RANDINVG generate random variates from the inverse Gaussian distribution.
%   out = randinvg(mu, lambda) generates random numbers from the 
%   inverse Gaussian distribution with parameters mu and lambda.
%
%   The input arguments are:
%       mu     - [n x 1] vector of the means
%       lambda - [1 x 1] scalar
%
%   Return values:
%       out     - [n x 1] vector of inverse Gaussian random variates 
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%% RNG uses alternative parameterisation
lambda = 1/lambda;
p = length(mu);

%% Do the sampling
V = randn(p,1).^2;
out = mu + 0.5*mu./lambda .* ( mu.*V - sqrt(4*mu.*lambda.*V + mu.^2.*V.^2) );

l = rand(p,1) >= mu./(mu+out);
out( l ) = mu(l).^2./out( l ); 

%% Hack to (hopefully) stop numerical issues!!! 
out = max(out, eps);

%%done
end
