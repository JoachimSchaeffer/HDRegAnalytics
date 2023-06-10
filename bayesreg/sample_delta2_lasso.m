function [delta2, delta2prod] = sample_delta2_lasso(b, sigma2, tau2, lambda2, delta2prod, delta2, groups, nGroups, GroupSizes)
%SAMPLE_DELTA2_LASSO samples the hyperparameters lambda2 for lasso model.
%   delta2 = sample_delta2_lasso(...) samples the delta2 group hyperparameters
%   from the conditional posterior distribution. Lasso prior for
%   group coefficients.
%
%   The input arguments are:
%       b          - [p x 1] regression coefficients
%       sigma2     - [1 x 1] noise variance
%       tau2       - [1 x 1] global variance hyperparameter
%       lambda2    - [p x 1] individual variance hyperparameters
%       delta2prod - [p x 1] product of group hyperparameters
%       delta2     - [ng x 1] group variance hyperparameters
%       groups     - [p x 1] group indices
%       nGroups    - [1 x 1] total number of groups in this layer
%       GroupSizes - [ng x 1] size of each group
%
%   Return values:
%       delta2     - a sample from the posterior distribution
%       delta2prod - updated product of group hyperparameters
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016


% Sample delta2 for each group
delta2prod = delta2prod ./ delta2(groups);
K = b.^2 ./ lambda2 ./ delta2prod;

for i = 1:nGroups
    ix = groups == i;
    
    % Sample the delta2 from the conditional
    ig_mu = sqrt( 2*tau2*sigma2 / sum(K(ix)) );
    delta2(i) = 1 / randinvg(ig_mu, 1/2);
    
    %gig_p = GroupSizes(i)/2 - 1;
    %gig_a = 1/tau2/sigma2 * sum(K(ix));
    %gig_b = 2;
        
    %delta2(i) = gigrnd(gig_p, gig_a, gig_b, 1);
end

delta2prod = delta2prod .* delta2(groups);

end