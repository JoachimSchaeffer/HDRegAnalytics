%PGDRAW Sample from the PolyaGamma distribution 
%   X = pgdraw(...) generates random variates from PG(1, z)
%
%   This file implements the Polya-gamma sampler PG(1,z).
%   This is a MATLAB implementation of Algorithm 6 in PhD thesis of Jesse 
%   Bennett Windle, 2013
%   URL: https://repositories.lib.utexas.edu/bitstream/handle/2152/21842/WINDLE-DISSERTATION-2013.pdf?sequence=1
%
%   The MATLAB implementation is vectorized as much as possible to
%   improve speed.
%
%   The input arguments are:
%       Z       - [N x 1] vector of scale parameters
%
%   Return values:
%       X      - [N x 1] vector of samples from PG(1,Z)
%
%   References:
%
%   Jesse Bennett Windle
%   Forecasting High-Dimensional, Time-Varying Variance-Covariance Matrices
%   with High-Frequency Data and Sampling P´olya-Gamma Random Variates for
%   Posterior Distributions Derived from Logistic Likelihoods  
%   PhD Thesis, 2013   
%
%   Damien, P. & Walker, S. G. Sampling Truncated Normal, Beta, and Gamma Densities 
%   Journal of Computational and Graphical Statistics, 2001, 10, 206-215
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2017
%

function X = pgdraw(z)

n = length(z);

% PG(b, z) = 0.25 * J*(b, z/2)
z = abs(z) / 2;

% Point on the intersection IL = [0, 4/ log 3] and IR = [(log 3)/pi^2, \infty)
t = 2 / pi;

%% computer the ratio q / (q + p)
const = z.^2/2 + pi^2/8;
logA = log(4) - log(pi) - z;
logK = log(const);
Kt = t * const;
w = 1/sqrt(t);

logf1 = logA + log(ncdf(w*(t*z - 1))) + logK + Kt;
logf2 = logA + 2*z  + log(ncdf(-w*(t*z+1))) + logK + Kt;
p_over_q = exp(logf1) + exp(logf2);
ratio = 1 ./ (1 + p_over_q); % q / (p + q)

%% setup variables for vectorisation
X = zeros(n, 1);
Isampled = false(n, 1);
Sn = zeros(n, 1);
u  = zeros(n, 1);

%% main sampling loop
while(~all(Isampled))

    %% Step 1: Sample X ? g(x|z)
    u(~Isampled) = rand(sum(~Isampled),1);

    %% Sample exponential, as required
    Ix = ~Isampled & (u < ratio);
    X(Ix) = t + exprnd_fast(ones(sum(Ix), 1))./const(Ix);
    
    %% Sampled from truncated inverse-gaussian, as required
    Ix = ~Isampled & ~(u < ratio);
    X(Ix) = truncinvgrng_vec(z(Ix), t);

    %% Step 2: Iteratively calculate Sn(X|z), starting at S1(X|z), until U <= Sn(X|z) for an odd n or U > Sn(X|z) for an even n
    i = 1;
    Ix = ~Isampled;
    Sn(Ix) = a(0, X(Ix), t);
    U = rand(n,1) .* Sn;

    even = false;
    sign = -1;
    
    Idone = Isampled;
    
    % While some RVs have either not been accepted, or have not yet been
    % rejected ...
    while(~all(Idone))
        Ix = ~Idone;
        Sn(Ix) = Sn(Ix) + sign*a(i, X(Ix), t);
        
        %Sn = Sn + sign * a(i, X, t);

        % Accept if n is odd
        Ix = ((U <= Sn) & ~even & ~Isampled);
        X(Ix) = X(Ix)/4;
        Isampled(Ix) = true;
        Idone(Ix) = true;
        
        % Return to step 1 if n is even
        Ix = (U > Sn) & ~Isampled & ~Idone & even;
        X(Ix) = X(Ix);
        Idone(Ix) = true;

        even = ~even;
        sign = -sign;
        i = i + 1;
    end
    
end

end


% Function a_n(x) defined in equations (12) and (13) of
% Bayesian inference for logistic models using Polya-Gamma latent variables
% Nicholas G. Polson, James G. Scott, Jesse Windle
% arXiv:1205.0310
%
% Also found in the PhD thesis of Windle (2013) in equations
% (2.14) and (2.15), page 24
function f = a(n, x, t)

f = zeros(length(x), 1);

Ix = x <= t;
f(Ix) = log(pi) + log(n + 0.5) + (3/2)*(log(2)-log(pi)-log(x(Ix))) - 2*(n + 0.5).^2./x(Ix);
Ix = x > t;
f(Ix)  = log(pi) + log(n + 0.5) - x(Ix) * pi^2 / 2 * (n + 0.5)^2;

f = exp(f);

end


%% Sample from a truncated inverse Gaussian
function X = truncinvgrng_vec(z, t)

mu = 1./z;
n = length(mu);
X = zeros(n, 1);

%% Rejection sampler based on truncated gamma
Ix = mu > t;
nx = sum(Ix);
Z = zeros(nx, 1);
Isampled = false(nx, 1);
zz = z(Ix);

while (~all(Isampled))
    u = rand(nx, 1);
    Z(~Isampled) = 1.0 ./ truncgamma_vec(sum(~Isampled), 1/t);
    
    Isampled(u < exp(-zz.^2/2 .* Z)) = true;
end
X(Ix) = Z;

%% Direct rejection sampler
Ix = ~Ix;
Isampled = false(sum(Ix), 1);
Z = zeros(sum(Ix), 1);
m = mu(Ix);
while (~all(Isampled))
    Z(~Isampled) = randinvg(m(~Isampled), 1);
    Isampled(Z < t) = true;
end
X(Ix) = Z;
    
end


% Sample truncated gamma random variates
%
%   Damien, P. & Walker, S. G. Sampling Truncated Normal, Beta, and Gamma Densities 
%   Journal of Computational and Graphical Statistics, 2001, 10, 206-215
function X  = truncgamma_vec(n, c)

X = zeros(n, 1);
Isampled = false(n, 1);

gX = zeros(n, 1);

while (~all(Isampled))
    X(~Isampled) = c + exprnd_fast(ones(sum(~Isampled),1))*2;
    gX(~Isampled) = sqrt(2/pi) ./ sqrt(X(~Isampled));%    exp(-(b-g)*X(~Isampled))/K./sqrt(X(~Isampled));
    
    Isampled(~Isampled) = (rand(sum(~Isampled),1) <= gX(~Isampled));
end

end


function f = ncdf(z)

f = 0.5 * erfc(-z ./ sqrt(2));

end
