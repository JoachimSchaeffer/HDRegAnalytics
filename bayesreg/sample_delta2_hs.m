function [delta2, delta2prod] = sample_delta2_hs(b, sigma2, tau2, lambda2, rho, delta2prod, delta2, groups, nGroups, GroupSizes)
%SAMPLE_DELTA2_HS samples the hyperparameters lambda2 for HS model.
%   delta2 = sample_delta2_hs(...) samples the delta2 group hyperparameters
%   from the conditional posterior distribution. Horseshoe prior for
%   group coefficients beta.
%
%   The input arguments are:
%       b          - [p x 1] regression coefficients
%       sigma2     - [1 x 1] noise variance
%       tau2       - [1 x 1] global variance hyperparameter
%       lambda2    - [p x 1] individual variance hyperparameters
%       rho        - [ng x 1] auxilliary hyperparameters
%       delta2prod - [p x 1] product of group hyperparameters
%       delta2     - [ng x 1] group variance hyperparameters
%       groups     - [p x 1] group indices
%
%   Return values:
%       delta2     - a sample from the posterior distribution
%       delta2prod - updated product of group hyperparameters
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

% Sample delta2 for each group
delta2prod = delta2prod ./ delta2(groups);

K = b.^2 ./ lambda2 ./ delta2prod;

%scale = 1./rho(1:end-1)' + 1/2/tau2/sigma2*sample_delta2_cpp(nGroups, K', groups);
%delta2(1:end-1) = scale ./ randg( (GroupSizes'+1)/2 );

% FASTER MATLAB APPROACH
scale = zeros(1, nGroups);
for i = 1:nGroups
   ix = groups == i;
   scale(i) = 1/rho(i) + 1/2/tau2/sigma2 * sum(K(ix));
end
delta2(1:end-1) = scale ./ randg( (GroupSizes'+1)/2 );

% for i = 1:nGroups
%     ix = groups == i;
%     
%     scale = 1/rho(i) + 1/2/tau2/sigma2 * sum(K(ix));
%     shape = (GroupSizes(i) + 1)/2;
%     
%     delta2(i) = scale / randg(shape,1);
% end

delta2prod = delta2prod .* delta2(groups);

end
