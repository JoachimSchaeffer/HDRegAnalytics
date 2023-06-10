%% Example 2
%  This example demonstrates how to use bayesreg to fit Bayesian linear
%  regression models and how to use br_summary() to display sampling
%  statistics.
clear;

fprintf('Example 2 - Fitting linear models to data\n');

%% Generate some data
n = 1e2;
p = 10;
snr = 4;
X = randn(n,p);
beta = ones(p,1);
mu = X*beta;
sigma2 = var(mu) / snr;
y = mu + randn(n,1)*sqrt(sigma2);

%% Bayesian linear regression (Gaussian error model, g-prior)
[b, b0, retval] = bayesreg(X,y,'gaussian','g','nsamples',1e4,'burnin',1e4,'thin',5,'display',false);

%% Print summary
br_summary(b,b0,retval);

%% Plots
% Names of variables
vn = cell(p,1);
for i = 1:p
    vn{i} =['v',num2str(i)];
end

% Plot regression coefficients samples
figure; 
boxplot(b',vn); grid;
title('Example 2');

% Plot sigma2 samples
figure; 
plot(retval.sigma2)
title('Example 2');
xlabel('Samples');
ylabel('\sigma^2');
