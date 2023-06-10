function nu = sample_nu_hsplus(lambda2, phi2)
%SAMPLE_NU_HSPLUS samples the hyperparameters nu for the HS+ model.
%   lambda2 = sample_nu_hsplus(...) samples the nu hyperparameters
%   from the conditional posterior distribution. Horseshoe+ prior for
%   regression coefficients beta.
%
%   The input arguments are:
%       lambda2 - [p x 1] hyperparameters
%       phi2    - [p x 1] hyperparameters
%
%   Return values:
%       nu      - [p x 1] sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

scale = 1./phi2 + 1./lambda2;
nu = 1 ./ exprnd_fast(1./scale);

end