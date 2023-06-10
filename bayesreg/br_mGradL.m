function [L, grad, H_b0, eta, extra_stats] = br_mGradL(y, X, theta, eta, model, xi, Xty)
%BR_MGRADL gradient and likelihoods for GLM models
%   [L, grad, H_b0, eta, extra_stats] = br_mGradL(...) computes and returns
%   likelihood and gradient information for GLM models
%
%   The input arguments are:
%       y      - [n x 1] targets
%       X      - [n x p] design matrix (predictors)
%       theta  - [p+1 x 1] stacked vector of [beta; beta0], with beta the
%                coefficients and beta0 the intercept
%       eta    - [n x 1] linear predictor (pass [] if not using)
%       model  - [1 x 1] target distribution (poisson, geometric, gamma, binomial, gaussian)
%       xi     - additional model parameters (scale for gamma)
%       Xty    - [n x 1] precomputed X'*y vector
%
%   Return values:
%       L           - [1 x 1] negative log-likelihood (up to constants)
%       grad        - [p+1 x 1] gradient of neg-log-likelihood wrt beta and beta0
%       H_b0        - [1 x 1] Hessian of neg-log-likelihood wrt to beta0
%       eta         - [n x 1] linear predictor for model
%       extra_stats - additional statistics for models (gamma)
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2020

extra_stats = [];

% Form the linear predictor if needed
if (isempty(eta))
    eta = X*theta(1:end-1) + theta(end);
end

% Poisson regression
if (strcmp(model,'poisson'))
    % Form the linear predictor
    eta = min(eta, 500);
    mu  = exp(eta);

    % Poisson likelihood (up to constants)
    L = sum(mu) - sum( y.*eta );

    % Poisson gradient
    grad = zeros(length(theta),1);
    grad(1:end-1) = X'*mu - Xty;
    grad(end)     = sum(mu) - sum(y);
    
    H_b0 = sum(mu);

% Geometric regression
elseif (strcmp(model,'geometric'))
    r = xi(1);
    
    eta = min(eta, 500);
    mu  = exp(eta);

    % Geometric likelihood (up to constants)
    L = -eta'*y + log(mu+r)'*(y+1);

    % Geometric gradient
    grad = zeros(length(theta),1);
    c = (mu.*(y+1)./(mu+r));
    grad(1:end-1) = X'*c - Xty;
    grad(end)     = sum(c) - sum(y);
    
    H_b0 = sum(mu./(mu+1));   
    
% Gaussian regression
elseif (strcmp(model,'gaussian'))
    e = (y-eta);
    L = sum(e.^2)/2;

    grad = zeros(length(theta),1);
    grad(1:end-1) = (-e'*X)';
    grad(end) = -sum(e);
    
    H_b0 = size(X,1);
    
% Binomial regression
elseif (strcmp(model,'binomial'))
    mu = eta;

    %% numerical constants
    lowerBnd = log(eps); 
    upperBnd = -lowerBnd;
    probLims = [eps, 1-eps];    

    %% Compute negative log-likelihood
    mu = constrain(mu, lowerBnd, upperBnd);
    prob = 1./(1 + exp(-mu));

    if any(prob(:) < probLims(1) | probLims(2) < prob(:))
        prob = max(min(mu,probLims(2)),probLims(1));
    end

    % Likelihood
    L = sum(-y.*log(prob) - (1-y).*log(1.0-prob));

    % Gradient
    grad = zeros(length(theta),1);
    grad(1:end-1) = (prob-y)'*X;
    grad(end)  = sum((prob-y));    

    H_b0 = sum(prob.*(1-prob));    

end

end

function x = constrain(x,lower,upper)

% Constrain between upper and lower limits, and do not ignore NaN
x(x<lower) = lower;
x(x>upper) = upper;

%% done;
end