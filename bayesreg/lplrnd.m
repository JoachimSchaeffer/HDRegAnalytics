function x = lplrnd(m, s, a, b)
%LPLRND generate a [a x b] random matrix from Laplace(m, s).
%   x = lplrnd(...) generates random numbers from the 
%   Laplace distribution with mean m and scale parameter s.
%
%   The input arguments are:
%       m    - [1 x 1] mean parameter 
%       s    - [1 x 1] scale parameter
%       a    - [1 x 1] size of output matrix
%       b    - [1 x 1] size of output matrix
%
%   Return values:
%       x     - [a x b] matrix of laplace(m,s) random variates
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

if(nargin < 4)
    b = 1;
end

x = rand(a, b) - 0.5;
x = m - s*sign(x) .* log(1 - 2*abs(x));

end
