function Z = polyexpand(x, k)
%POLYEXPAND Expand a vector into a matrix of Legendre polynomials
%  Z = polyexpand(x, k) expands the vector x into a k-th degree Legendree
%  polynomial. The constant column is removed from the expansion matrix.
%  The vector x must be [-1, 1].
%       x          - [n x 1] data vector in the range [-1,+1]
%       k          - [1 x 1] order of polynomial (k>=2)
%
%
%   Return values:
%       Z          - [n x k+1] matrix of Legendre polynomials
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

n = length(x);
Z = zeros(n, k+1);

%% Legendre polynomials
Z(:,1) = ones(n, 1);
Z(:,2) = x; 
for j = 2:k
    N = j - 1;
    Z(:,j+1) = ( (2*N + 1) * x .* Z(:,j) - N*Z(:,j-1) ) ./ (N + 1);
end

%% Remove the constant column
Z(:,1) = [];

end