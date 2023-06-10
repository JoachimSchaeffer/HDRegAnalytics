%% Bayesian Linear Regression to Analyze the LFP Data
clear variables 
close all
clc

%% Load LFP Data 
lfp_data = readmatrix("data/lfp_slim.csv", "NumHeaderLines", 1);
X = flip(lfp_data(:, 2:1001), 2);
X_ = X - mean(X, 1);
y = lfp_data(:, 1002);
x_lfp = linspace(2.0, 3.5, 1000);

train_id = lfp_data(:, 1004) == 0;

X_train = X(train_id, :);
y_train_mean = mean(X_train, 2);
X_train_ = X_train - mean(X_train, 1);
y_train_mean_ = y_train_mean - mean(y_train_mean);

%% Data Visualization
figure;
plot(x_lfp, X)

figure;
plot(x_lfp, X_train)

figure; 
plot(x_lfp, X_train_)
% X is fine. 

beta_true = 1/size(X, 2) *ones(size(X, 2));

%% Thinning: 
% https://stats.stackexchange.com/questions/485859/what-value-of-thinning-is-acceptable-in-bayesian-data-analysis
% https://stats.stackexchange.com/questions/442714/why-does-thinning-work-in-bayesian-inference
% Thinning has nothing to do with Bayesian analysis, it is concerned with 
% the convergence of an MCMC sequence. Without thinning, correlation 
% between the points may slow down convergence or worse give the impression 
% of convergence while the chain has not visited the entire space. 
% The thinning factor intends to reproduce iid sampling
% Higher values for thinnin will increase the run time.

%% Burnin: 
% https://www.johndcook.com/blog/2016/01/25/mcmc-burn-in/
% Important for MCMC sampling to ensure being in an area of high
% probablilty as this is not known a priori.

%% Bayesregression
% For functional data:
% Relevant noise models: 'gaussian','laplace','t'
% Relevant priors: 'ridge','lasso','horseshoe' 
% Future experiments could see how the fused lasso compares with ridge
% regression (fused lasso penalizes for regression coefficient differeces).


std_bool = [true, false];

for i=1:2
    [b, b0, retval] = bayesreg( ...
        X_train_, ...
        y_train_mean_, ...
        't', ...
        'ridge', ...
        'nsamples',1e4, ...
        'burnin',1e3, ...
        'thin',1, ...
        'display',false, ...
        'normalize', std_bool(i), ...
        'tau2prior', 'uniform' ...
    );
    
    % Plot the results
    % Make a figure that resembles the figure in the notebook. 
    figure; 
    mean_b = mean(b,2);
    std_b = std(b, 0, 2);
    
    %% Dimensions currenlty wrong!
    plot(mean_b')
    hold on
    plot(mean_b'+std_b')
    plot(mean_b'-std_b')
    hold on
    plot(beta_true)
    figure; 
    plot(retval.tStat)

end
% T-test recovers stadnard deviation of data, for tjhe standardized testm,
% cool! 

%%  Plot sigma2 samples
figure; 
plot(retval.tStat)

figure; 
plot((mean_b./std_b)')

%% TODO List
% Apply BLR on the real battery data
% See the distributions of shrinkage parameters
% Change priors (maybe 0.1 instead of 1?)
% Hack way of incorporating SNR

% Finalize the Bayesian Anlaysis, ghenerate data and make figures in python
% for the SI. 

% 