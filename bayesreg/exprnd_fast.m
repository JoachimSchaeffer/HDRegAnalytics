function r = exprnd_fast(mu)
%EXPRND_FAST generate random variates from the exponential distribution.
%   r = exprnd_fast(mu) generates random numbers from the 
%   exponential distribution with mean mu.
%
%   The input arguments are:
%       mu    - [n x 1] vector of the the means
%
%   Return values:
%       r     - [n x 1] vector of exponential random variates with mean mu
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%% length of mu
n = length(mu);

%% Generate uniform random values, and apply the exponential inverse CDF.
r = -mu .* log(rand(n, 1));

end