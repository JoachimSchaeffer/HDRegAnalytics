function x = igamrnd(alpha, beta)
%IGAMRND draws samples from an inverse gamma distribution
%   x = igamrnd(alpha, beta) samples from an inverse gamma distribution with
%   shape alpha and scale beta.
%
%   The input arguments are:
%       alpha    - [1 x 1] shape parameter 
%       beta     - [n x 1] scale parameters
%
%   Return values:
%       x        - [n x 1] inverse gamma samples
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2019

x = beta ./ randg(alpha, length(beta), 1);

end
