function varranks = bfr(b)
%BFR computes the rank of each predictor using the BFR algorithm.
%
%   The input arguments are:
%       beta    - [p x NSAMPLES] regression parameters
%
%   Return values:
%       ranks   - [p x 1]
%
%   References:
%     Makalic, E. & Schmidt, D. F. 
%     A Simple Bayesian Algorithm for Feature Ranking in High Dimensional Regression Problems 
%     24th Australasian Joint Conference on Advances in Artificial
%     Intelligence (AIA 2011), Vol. 7106, 223-230, 2011
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%%
[p, nsamples] = size(b);
ranks = zeros(p, nsamples);


for i = 1:nsamples
    [~,O]=sort(abs(b(:,i)),'descend');
    ranks(O,i) = 1:p;
end

%% Determine the 75th percentile
q = prctile(ranks', 75)';
[~,O] = sort(q);

%% Determine ranking with ties
varranks = nan(p+1,1);
j = 1;
k = 1;
for i = 1:p
    if (i >= 2)
        if (q(O(i)) ~= q(O(i-1)))
            j = j+k;
            k = 1;
        else
            k = k+1;
        end
    end
    varranks(O(i)) = j;
end

end