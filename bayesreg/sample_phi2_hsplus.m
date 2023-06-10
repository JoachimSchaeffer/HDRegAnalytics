function phi2 = sample_phi2_hsplus(nu, zeta)
%SAMPLE_PHI2_HSPLUS samples the hyperparameters phi2 for the HS+ prior.
%   phi2 = sample_phi2_hsplus(...) samples the phi2 hyperparameters
%   from the conditional posterior distribution. HS+ prior.
%
%   The input arguments are:
%       nu      - [p x 1] hyperparameters
%       zeta    - [p x 1] hyperparameters
%
%   Return values:
%       phi2    - [p x 1] sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

scale = 1./nu + 1./zeta;
phi2 = 1 ./ exprnd_fast(1./scale);

end