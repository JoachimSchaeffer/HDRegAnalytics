    function [b, L_b, grad_b, eta, H_b0, A, extra_stats] = br_mGradSampleBeta(b, b0, L_b, grad_b, D, delta, eta, H_b0, y, X, sigma2, Xty, model, extra_model_params, extra_stats)
%MH_TUNE update tuning parameters for a Metropolis-Hastings sampler
%   tune = mh_Tune(...) updates the adaptive step-size for a 
%   Metropolis-Hastings sampler
%
%   The input arguments are:
%       tune   - [1 x 1] a Metropolis-Hastings tuning structure
%
%   Return values:
%       tune   - [1 x 1] updated Metropolis-Hastings tuning structure
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2020

[~, px] = size(X);

% Get quantities need for proposal
[mGrad_prop_1, mGrad_prop_2] = mGrad_update_proposal(px, D, delta);

% Generate proposal from marginal proposal distribution    
bnew = normrnd( mGrad_prop_2.*((2/delta)*(b - (delta/2)*grad_b/sigma2)), sqrt((2/delta)*mGrad_prop_1+mGrad_prop_2) );

% Accept/reject?
extra_stats_new = [];
switch(model)
    case 'poisson'
        %[L_bnew,grad_bnew,~,eta_new,H_b0new] = mGrad_poissL(y, X, bnew, b0, Xty);
        [L_bnew, grad_bnew, H_b0_new, eta_new] = br_mGradL(y, X, [bnew;b0], [], 'poisson', extra_model_params, Xty);
    case 'geometric'
        %[L_bnew,grad_bnew,~,eta_new,H_b0new] = mGrad_geoL(y, X, bnew, b0, Xty);
        [L_bnew, grad_bnew, H_b0_new, eta_new] = br_mGradL(y, X, [bnew;b0], [], 'geometric', extra_model_params, Xty);
    %case 'gaussian'
    %    [L_bnew,grad_bnew,~,eta_new,H_b0new] = mGrad_gaussianL(y, X, bnew, b0);
    %case 'binomial'
    %    [L_bnew,grad_bnew,~,eta_new,H_b0new] = mGrad_binomialL(y, X, bnew, b0);
    %case 'gamma'
    %    [L_bnew,grad_bnew,~,eta_new,H_b0new,extra_stats_new] = mGrad_gammaL(y, X, bnew, b0);
end
grad_bnew = grad_bnew(1:end-1);

h1 = mGrad_hfunc(b, bnew, -grad_bnew/sigma2, mGrad_prop_2, delta);
h2 = mGrad_hfunc(bnew, b, -grad_b/sigma2, mGrad_prop_2, delta);
 
mhprob = min(exp(-L_bnew/sigma2 - -L_b/sigma2 + h1 - h2), 1);

A = false;
if (rand < mhprob && ~any(isinf(bnew)))
    A = true;
    b = bnew;
    L_b = L_bnew;
    grad_b = grad_bnew;
    eta = eta_new;
    H_b0 = H_b0_new;
    extra_stats = extra_stats_new;
end        

end


%% 'h'-function for mGrad MH acceptance probability
function h = mGrad_hfunc(x, y, grad_y, v, delta)

h = ((x - (2/delta)*v.*(y + (delta/4)*grad_y)).*(1./((2/delta)*v+1)))'*grad_y;

end


%% Update the quantities used for MH proposal
function [vp1, vp2] = mGrad_update_proposal(p, Lambda, delta)

% Update proposal information
vp2 = ones(p,1).*(1./(1./Lambda + (2/delta)));
vp1 = vp2.^2;

end


%%
% =========================================================================
%  Likelihood/Gradient functions
%   Poisson, Gaussian
% =========================================================================

%% Poisson likelihood and gradient
function [L, grad, grad_b0, eta, H_b0] = mGrad_poissL(y, X, b, b0, Xty)

eta = X*b(1:end) + b0;
eta = min(eta, 500);
mu  = exp(eta);

% Poisson likelihood (up to constants)
L = sum(mu) - sum( y.*eta );

% Poisson gradient
if (nargout > 1)
    grad = zeros(length(b),1);
    grad(1:end) = X'*mu - Xty;
    grad_b0 = sum(mu) - sum(y);
end

% Hessian for b0
H_b0 = sum(mu);

end

function [L, grad, H_b0, extra_stats] = mGrad_poissL_eta(y, X, eta, Xty)

mu  = exp(eta);

% Poisson likelihood (up to constants)
L = sum(mu) - sum( y.*eta );

% Poisson gradient
grad = zeros(length(Xty),1);
grad(1:end) = X'*mu - Xty;

% Hessian for b0
H_b0 = sum(mu);

extra_stats = [];

end

%% Geometric likelihood and gradient
function [L, grad, grad_b0, eta, H_b0] = mGrad_geoL(y, X, b, b0, Xty)

r = 1;

eta = X*b(1:end) + b0;
eta = min(eta, 500);
mu  = exp(eta);

% Geometric likelihood (up to constants)
L = -eta'*y + log(mu+r)'*(y+1);

% Geometric gradient
grad = zeros(length(b),1);
c = (mu.*(y+1)./(mu+r));
grad(1:end) = X'*c - Xty;
grad_b0       = sum(c) - sum(y);

% Hessian for b0
H_b0 = sum(mu./(mu+1));

end

function [L, grad, H_b0, extra_stats] = mGrad_geoL_eta(y, X, eta, Xty)

r = 1;

mu  = exp(eta);

% Geometric likelihood (up to constants)
L = -eta'*y + log(mu+r)'*(y+1);

% Geometric gradient
grad = zeros(size(X,2),1);
c = (mu.*(y+1)./(mu+r));
grad(1:end) = X'*c - Xty;

% Hessian for b0
H_b0 = sum(mu./(mu+1));

extra_stats = [];

end

%% Gaussian likelihood and gradient (with sigma2 = 1)
function [L, grad, grad_b0, eta, H_b0] = mGrad_gaussianL(y, X, b, b0)

eta = X*b + b0;
e = (y-eta);
L = sum(e.^2)/2;

grad = (-e'*X)';
grad_b0 = 0;

H_b0 = size(X,1);

end

function [L, grad, H_b0, extra_stats] = mGrad_gaussianL_eta(y, X, eta)

e = (y-eta);
L = sum(e.^2)/2;

grad = (-e'*X)';

H_b0 = size(X,1);

extra_stats = [];

end

%% Gamma likelihood and gradient (with sigma2 (kappa) = 1)
function [L, grad, grad_b0, eta, H_b0, extra_stats] = mGrad_gammaL(y, X, b, b0)

eta = X*b + b0;
mu  = exp(eta);
mu = min(mu, 1e8);
mu = max(mu, 1e-8);

% % Gamma neg-log-likelihood (up to constants)
y_on_mu = y./mu;
sum_y_on_mu = sum(y_on_mu);
sum_eta = sum(eta);
L = sum_eta + sum_y_on_mu;

% Gamma gradient
grad = -sum( bsxfun(@times, (y_on_mu - 1), X), 1 )';
grad_b0 = -sum_y_on_mu + length(y);

H_b0 = sum_y_on_mu;

% Return additional statistics required for re-computing likelihoods quickly
extra_stats = [sum_eta, sum_y_on_mu];

end

function [L, grad, H_b0, extra_stats] = mGrad_gammaL_eta(y, X, eta)

mu  = exp(eta);

% Gamma neg-log-likelihood (up to constants)
y_on_mu = y./mu;
sum_y_on_mu = sum(y_on_mu);
sum_eta = sum(eta);
L = sum_eta + sum_y_on_mu;

% Gamma gradient
grad = -sum( bsxfun(@times, (y_on_mu - 1), X), 1 )';

H_b0 = sum_y_on_mu;

extra_stats = [sum_eta, sum_y_on_mu];

% % Gamma neg-log-likelihood (up to constants)
% L = sum( eta + y./mu );
% 
% % Gamma gradient
% grad = -sum( bsxfun(@times, (y./mu - 1), X), 1 )';
% 
% H_b0 = sum( y./mu );

end

%% Binomial likelihood and gradient
function [L, grad, grad_b0, eta, H_b0] = mGrad_binomialL(y, X, b, b0)

[~, px] = size(X);
eta = X*b + b0;

%% numerical constants
probLims = [eps, 1-eps];    

%% Compute negative log-likelihood
prob = 1./(1 + exp(-eta));

if any(prob(:) < probLims(1) | probLims(2) < prob(:))
    prob = max(min(eta,probLims(2)),probLims(1));
end

% Likelihood
L = sum(-y.*log(prob) - (1-y).*log(1.0-prob));

% Gradient
grad = zeros(length(b),1);
grad(1:px) = (prob-y)'*X;
%grad_b0  = sum((prob-y));
grad_b0 = 0;

% Hessian for intercept
H_b0 = sum(prob.*(1-prob));

end

function [L, grad, H_b0, extra_stats] = mGrad_binomialL_eta(y, X, eta, extra_stats)

[~, px] = size(X);

%% numerical constants
probLims = [eps, 1-eps];    

%% Compute negative log-likelihood
prob = 1./(1 + exp(-eta));

if any(prob(:) < probLims(1) | probLims(2) < prob(:))
    prob = max(min(eta,probLims(2)),probLims(1));
end

% Likelihood
L = sum(-y.*log(prob) - (1-y).*log(1.0-prob));

% Gradient
grad = zeros(size(X,2),1);
grad(1:px) = (prob-y)'*X;

% Hessian for intercept
H_b0 = sum(prob.*(1-prob));

extra_stats = [];

end

function x = constrain(x,lower,upper)

% Constrain between upper and lower limits, and do not ignore NaN
x(x<lower) = lower;
x(x>upper) = upper;

%% done;
end
