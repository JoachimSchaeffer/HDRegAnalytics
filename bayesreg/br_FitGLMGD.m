function [b, b0, L, gradb, gradb0] = br_FitGLMGD(X, y, model, xi, tau2, maxiter)
%BR_FITGLMGD simple gradient descent fitting of generalized linear models
%   br_FitGLMGD(...) fits a generalized linear model to data using the
%   specified target distribution
%
%   The input arguments are:
%       X       - [n x p] matrix of predictors
%       y       - [n x 1] vector of targets
%       model   - string, one of {'poisson','geometric','exp','gamma','igauss','gaussian','binomial'}
%       xi      - additional model parameters for gamma, i-gaussian
%       tau2    - global shrinkage parameter (0=max regularisation, inf=no regularisation)
%       maxiter - maximum number of iterations to run GD for
%
%   Return values:
%       b           - [p x 1] estimates of beta
%       b0          - [1 x 1] estimate of intercept
%       L           - [1 x 1] negative log-likelihood at estimates (up to constants)
%       gradb       - [p x 1] gradient wrt b at estimates
%       gradb0      - [1 x 1] gradient wrt b0 at estimates
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2019

if (~exist('maxiter','var'))
    maxiter = 5e2;
end

%% Initialisation
nx    = size(X,1);
px    = size(X,2);
theta = zeros(px+1, 1);

% Precompute statistics
Xty = X'*y;

% Learning rate
kappa = 1;

%% Error checking
model = validatestring(model, {'poisson','geometric','exp','gamma','igauss','binomial','gaussian'});
if (strcmp(model,'poisson') || strcmp(model,'geometric'))
    if (min(y) < 0 || ~all(floor(y) == y))
        error('Data for Poisson/geometric regression must be non-negative integers (counts)');
    end
elseif (strcmp(model,'exp') || strcmp(model,'gamma') || strcmp(model,'igauss'))
    if (min(y) < 0)
        error('Data for exponential/gamma/inverse-Gaussian regression must be non-negative real numbers');
    end
end    

if (strcmp(model,'binomial'))
    if ((sum(y==0) + sum(y==1)) ~= nx)
        error('Data for binomial regression must consist only of 0s and 1s');
    end
end

if (strcmp(model,'gamma') || strcmp(model,'igauss'))
    if (length(xi) ~= 1 || xi(1) < 0)
        error('Gamma/inverse-Gaussian regression requires ''xi'' be a non-negative real number.');
    end
end

% Any special initialisation
if (strcmp(model,'gaussian'))
    theta(end) = mean(y);
end

%% Optimise
[L, grad] = gradL(y, X, theta, model, Xty, xi, tau2);
for i = 1:maxiter
    % Update estimates
    kappa_vec = ones(px+1, 1)*kappa;
    kappa_vec(end) = kappa_vec(end)/sqrt(nx);
    theta_new = theta - kappa_vec.*grad(1:end);

    % Have we improved?
    [L_new, grad_new] = gradL(y, X, theta_new, model, Xty, xi, tau2);
    if (L_new < L)
        theta = theta_new;
        L = L_new;
        grad = grad_new;
    else
        % If not, halve the learning rate
        kappa = kappa/2;
    end
end

%% Return
b      = theta(1:end-1);
b0     = theta(end);
gradb  = grad(1:end-1);
gradb0 = grad(end);

end

%% Likelihood/gradient function
function [L, grad] = gradL(y, X, theta, model, Xty, xi, tau2)

% Poisson regression
if (strcmp(model,'poisson'))
    % Form the linear predictor
    eta = X*theta(1:end-1) + theta(end);
    eta = min(eta, 500);
    mu  = exp(eta);

    % Poisson likelihood (up to constants)
    L = sum(mu) - sum( y.*eta ) + sum(theta(1:end-1).^2)/2/tau2;

    % Poisson gradient
    grad = zeros(length(theta),1);
    grad(1:end-1) = X'*mu - Xty + theta(1:end-1)./tau2;
    grad(end)   = sum(mu) - sum(y);

% Geometric regression
elseif (strcmp(model,'geometric'))
    r = xi(1);
    
    % Form the linear predictor
    eta = X*theta(1:end-1) + theta(end);
    eta = min(eta, 500);
    mu  = exp(eta);

    % Geometric likelihood (up to constants)
    L = -eta'*y + log(mu+r)'*(y+1) + sum(theta(1:end-1).^2)/2/tau2;

    % Geometric gradient
    grad = zeros(length(theta),1);
    c = (mu.*(y+1)./(mu+r));
    grad(1:end-1) = X'*c - Xty + theta(1:end-1)/tau2;
    grad(end)     = sum(c) - sum(y);
       
% Gaussian regression
elseif (strcmp(model,'gaussian'))
    eta = X*theta(1:end-1) + theta(end);
    e = (y-eta);
    L = sum(e.^2)/2 + sum(theta(1:end-1).^2)/2/tau2;

    grad = zeros(length(theta),1);
    grad(1:end-1) = (-e'*X)' + theta(1:end-1)/tau2;
    grad(end) = -sum(e);
    
% Binomial regression
elseif (strcmp(model,'binomial'))
    mu = X*theta(1:end-1) + theta(end);

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
    L = sum(-y.*log(prob) - (1-y).*log(1.0-prob)) + sum(theta(1:end-1).^2)/2/tau2;

    % Gradient
    grad = zeros(length(theta),1);
    grad(1:end-1) = (prob-y)'*X + theta(1:end-1)'/tau2;
    grad(end)  = sum((prob-y));    
    
end

end

function x = constrain(x,lower,upper)

% Constrain between upper and lower limits, and do not ignore NaN
x(x<lower) = lower;
x(x>upper) = upper;

%% done;
end