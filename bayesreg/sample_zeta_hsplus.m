function zeta = sample_zeta_hsplus(phi2)
%SAMPLE_ZETA_HSPLUS samples the hyperparameters zeta for HS+ prior.
%   zeta = sample_zeta_hsplus(...) samples hyperparameters zeta
%   from the conditional posterior distribution. All models.
%
%   The input arguments are:
%       phi2   - [p x 1] hyperparameters
%
%   Return values:
%       zeta   - [p x 1] sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

scale = 1 + 1./phi2;
zeta = 1 ./ exprnd_fast(1./scale);


end