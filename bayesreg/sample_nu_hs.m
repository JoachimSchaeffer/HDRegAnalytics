function nu = sample_nu_hs(lambda2)
%SAMPLE_NU_HS samples the hyperparameters nu for the HS model.
%   lambda2 = sample_nu_hs(...) samples the nu hyperparameters
%   from the conditional posterior distribution. Horseshoe prior for
%   regression coefficients beta.
%
%   The input arguments are:
%       nu      - [p x 1] hyperparameters
%
%   Return values:
%       nu      - [p x 1] sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

scale = 1 + 1./lambda2;
nu = 1 ./ exprnd_fast(1./scale);

end