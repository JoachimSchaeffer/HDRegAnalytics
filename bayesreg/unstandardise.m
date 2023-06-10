function [X, y] = unstandardise(X, muX, normX, y, muy)
%UNSTANDARDISE Remove effects of standardisation on input data.
%   [X, y] = unstandardise(X, muX, normX, y, muy) unstandardises the 
%   predictor matrix X and (optional) the target vector y. 
%
%   The input arguments are:
%       X     - matrix of size [n x p] to be unstandardised
%       muX   - original mean of X
%       normX - original length of each column of X
%       y     - vector of size [n x 1] to be unstandardised
%       muy   - original mean of y
%
%   Return values:
%       X     - the data matrix in the original scale
%       y     - the target vector in the original scale
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%% Unstandardise Xs
X=bsxfun(@times,X,normX);
X=bsxfun(@plus,X,muX);

%% Standardise ys (if neccessary)
if(nargin == 2)
    y = y + muy;
end;

end
