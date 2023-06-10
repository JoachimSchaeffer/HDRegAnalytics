%% Example 16
%  This example demonstrates how to use br_sparsify() to produce sparse
%  estimates of the regression coefficients following MCMC sampling with
%  bayesreg.
clear; rng(3157);

fprintf('Example 16 - Demonstration of count regression models using adaptive shrinkage priors\n');

%% Load the data 
n = 100;    % sample size
[T, ~, X, ~, ~, y] = br_create_example_table(n);

%% Fit models
% Bayesian Poisson regression using adaptive log-t
[b, b0, retval_poiss] = bayesreg(X,y,'poisson','logt','nsamples',1e4,'burnin',1e4,'thin',5,'display',true,'catvars',[4,7]);

% Bayesian geometric regression using adaptive log-t
[b, b0, retval_geo] = bayesreg(X,y,'geometric','logt','nsamples',1e4,'burnin',1e4,'thin',5,'display',true,'catvars',[4,7]);

% Use WAIC to decide which model fits better
fprintf('\n');
fprintf('Possion or Geometric?\n');
fprintf(' WAIC Poisson = %.3f vs WAIC Geometric = %.3f\n', retval_poiss.modelstats.waic, retval_geo.modelstats.waic);
fprintf('  => so we would prefer the Poisson regression.\n');
fprintf(' In this case the data was generated from a Poisson regression, so this is correct.\n');
fprintf(' * Also note the overdispersion for the Poisson is close to one suggesting a reasonable fit.\n');