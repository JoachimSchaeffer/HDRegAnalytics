function omega2 = sample_omega2_tdist(e, sigma2, tdof)
%SAMPLE_OMEGA2_TDIST samples the hyperparameters omega2 for the Student-t error model.
%   omega2 = sample_omega2_tdist(...) samples the omega2 hyperparameters
%   from the conditional posterior distribution. Student-t error model.
%
%   The input arguments are:
%       e       - [n x 1] errors
%       sigma2  - [1 x 1] residual variance
%       tdof    - [1 x 1] deegrees of freedom
%
%   Return values:
%       omega2  - [n x 1] sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

n = length(e);
a = (tdof + 1) / 2;
b = e.^2/sigma2/2 + tdof/2;

omega2 = b ./ randg(a, n, 1);

end