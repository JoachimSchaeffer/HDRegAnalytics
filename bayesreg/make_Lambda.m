function [Lambda, delta2prod] = make_Lambda(sigma2, tau2, lambda2, groups, delta2)
%MAKE_LAMBDA forms the diagonal entries of the Lambda regularisation matrix
%
%   The input arguments are:
%       sigma2   - [1 x 1] noise scale parameter
%       tau2     - [1 x 1] global variance hyperparameter
%       lambda2  - [p x 1] local variance hyperparameters
%       groups   - {ngroups x 1} cell array of vectors of groupings (one
%                  for each level)
%       delta2   - {ngroups x 1} cell array of vectors of group variance hyperparameters
%                  (one for each level)
%
%   Return values:
%       Lambda     - [p x 1] diagonal elements of the "Lambda" matrix
%       delta2prod - [p x 1] combined group variance hyperparameters
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2017

% Build group variance hyperparameters
delta2prod = ones(length(lambda2), 1);
if (~isempty(groups))
    for j = 1:length(groups)
        delta2prod = delta2prod .* (delta2{j}(groups{j}));
    end
end

% Build diagonal elements of regularisation matrix
Lambda = sigma2 * tau2 * lambda2 .* delta2prod;

end