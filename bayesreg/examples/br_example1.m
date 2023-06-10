%% Example 1 
%  This example demonstrates how to use bayesreg to fit Bayesian linear
%  regression models and how to use br_predict() to make predictions
%  from the fitted models.
clear;
fprintf('Example 1 - Fitting simple linear models to data\n');

%% Data
X = (1:10)';
y = [-0.6867 1.7258 1.9117 6.1832 5.3636 ...
    7.1139 9.5668 10.0593 11.4044 6.1677]';

%% Fit models
% Bayesian linear regression (Student t error model, lasso prior)
[t_beta, t_beta0, t_stats] = bayesreg(X,y,'t','lasso','nsamples',5e3,'burnin',1e3,'thin',5,'tdof',5,'display',false);

% Bayesian linear regression (Gaussian error model, lasso prior)
[g_beta, g_beta0, g_stats] = bayesreg(X,y,'gaussian','lasso','nsamples',5e3,'burnin',1e3,'thin',5,'display',false);

%% Compute predictions
fprintf('----------------------------------------------------------------------\n');
[pred_t, predstats_t] = br_predict(X, t_beta, t_beta0, t_stats, 'ytest', y, 'CI', [2.5, 97.5], 'display', true);

fprintf('----------------------------------------------------------------------\n');
[pred_gauss, predstats_gauss] = br_predict(X, g_beta, g_beta0, g_stats, 'ytest', y, 'CI', [2.5, 97.5], 'display', true);

%% Do some plotting
plot(X, y,'.','markersize',18);
grid;
hold on;
title('Example 1');

plot(X, pred_gauss{:,'yhat'}, 'k-');
plot(X, pred_gauss{:,'yhat_CI2_5'}, 'k--');
plot(X, pred_gauss{:,'yhat_CI97_5'}, 'k-.');

plot(X, pred_t{:,'yhat'}, 'r-');
plot(X, pred_t{:,'yhat_CI2_5'}, 'r--');
plot(X, pred_t{:,'yhat_CI97_5'}, 'r-.');

legend('Data', 'Gaussian', 'Gaussian (CI 2.5)', 'Gaussian (CI 97.5)', 'Student t (\nu = 5)', 'Student t (CI 2.5)', 'Student t (CI 97.5)', 'location', 'northwest');
xlabel('X');
ylabel('y');
hold off;
