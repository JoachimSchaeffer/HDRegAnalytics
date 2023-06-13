%% Bayesian Linear Regression to Analyze the LFP Data
clear variables 
close all
clc

addpath(genpath('bayesreg'))

synthethic_response_analysis = false;
cycle_life_response_snr_scaling = true; 

cycle_life_response_snr_db_scaling = true; 
cycle_life_response = true;

%% Load LFP Data 

lfp_data = readmatrix("data/lfp_slim.csv", "NumHeaderLines", 1);
X = flip(lfp_data(:, 2:1001), 2);
X_ = X - mean(X, 1);
% y = factor_y*lfp_data(:, 1002);
x_lfp = linspace(2.0, 3.5, 1000);

train_id = lfp_data(:, 1004) == 0;
% train_id(42) = 0;
test1_id = lfp_data(:, 1004) == 1;
test1_id(43) = 0; % Remove lowest cycle life battery!
test2_id = lfp_data(:, 1004) == 2;

X_train = X(train_id, :);
X_test1 = X(test1_id, :);
X_test2 = X(test2_id, :);

X_train_ = X_train - mean(X_train, 1);
X_test1_ = X_test1 - mean(X_train, 1);
X_test2_ = X_test2 - mean(X_train, 1);

y_train_mean = mean(X_train, 2);
y_train_mean_ = y_train_mean - mean(y_train_mean);

y = lfp_data(:, 1002);
y_train = log(y(train_id));
[y_train_, c_y_cl, s_y_cl] = normalize(y_train);
y_test1 = log(y(test1_id));
% y_test1_ = normalize(y_test1, "center", c_y_cl, "scale", s_y_cl);
y_test2 = log(y(test2_id));
% y_test2_ = normalize(y_test2, "center", c_y_cl, "scale", s_y_cl);


%% Data Visualization
figure;
plot(x_lfp, X)

figure;
plot(x_lfp, X_train)

figure; 
plot(x_lfp, X_train_)
% X is fine. 

beta_true = 1/size(X, 2) *ones(size(X, 2));

%% Bayesregression
% For functional data:
% Relevant noise models: 'gaussian','laplace','t'
% Relevant priors: 'ridge','lasso','horseshoe' 
% Future experiments could see how the fused lasso compares with ridge
% regression (fused lasso penalizes for regression coefficient differeces).
% Thinning: 
% https://stats.stackexchange.com/questions/485859/what-value-of-thinning-is-acceptable-in-bayesian-data-analysis
% https://stats.stackexchange.com/questions/442714/why-does-thinning-work-in-bayesian-inference
% Thinning has nothing to do with Bayesian analysis, it is concerned with 
% the convergence of an MCMC sequence. Without thinning, correlation 
% between the points may slow down convergence or worse give the impression 
% of convergence while the chain has not visited the entire space. 
% The thinning factor intends to reproduce iid sampling
% Higher values for thinnin will increase the run time.
% Burnin: 
% https://www.johndcook.com/blog/2016/01/25/mcmc-burn-in/
% Important for MCMC sampling to ensure being in an area of high
% probablilty as this is not known a priori.

if synthethic_response_analysis
    % Pathologisches Beispiel! 
    disp("Synthethic Response:")
    [b_std, b0_std, retval_std] = bayesreg_wrap(X_train_, y_train_mean_, true); 
    [mean_b_std, std_b_std] = plot_bayesian_reg_coef(b_std, retval_std, beta_true);
    [b, b0, retval] = bayesreg_wrap(X_train_, y_train_mean_, false); 
    [mean_b, std_b] = plot_bayesian_reg_coef(b, retval, beta_true);
    
    % T-test recovers standard deviation of data, for the standardized testm,
    % cool!  Plot sigma2 samples
    % figure; 
    % plot((mean_b./std_b)')
end

%% Apply BLR on the real battery data
if cycle_life_response
    disp("Cycle Life Response:")
    % [b_cl_std, b0_cl_std, retval_cl_std] = bayesreg_wrap(X_train_, y_train_, true); 
    % [mean_b_std, std_b_std] = plot_synthetic_example(b_cl_std, retval_cl_std, x_lfp);
    
    [b_cl, b0_cl, retval_cl] = bayesreg_wrap(X_train_, y_train_, false); 
    [mean_b_std, std_b_std] = plot_bayesian_reg_coef(b_cl, retval_cl, x_lfp);
    predict(X_train_, X_test1_, X_test2_, y_train, y_test1, y_test2, c_y_cl, s_y_cl, b_cl, true)
    
    [X_train_std, c_X_std_cl, s_X_std_cl] = normalize(X_train_);
    X_test1_std = normalize(X_test1_, "center", c_X_std_cl, "scale", s_X_std_cl);
    X_test2_std = normalize(X_test2_, "center", c_X_std_cl, "scale", s_X_std_cl);  
    figure; 
    plot(x_lfp, X_train_std, "Color", "black")
    hold on
    plot(x_lfp, X_test1_std, "Color", "blue")
    plot(x_lfp, X_test2_std, "Color", "red")
    [b_std_cl, b0_std_cl, retval_std_cl] = bayesreg_wrap(X_train_std, y_train_, false); 
    [mean_b_std, std_b_std] = plot_bayesian_reg_coef(b_std_cl, retval_std_cl, x_lfp);
    predict(X_train_std, X_test1_std, X_test2_std, y_train, y_test1, y_test2, c_y_cl, s_y_cl, b_std_cl, true)
    % Why does standardization still helps? Besides blowing up the noise
    % and feeding noise into the model, something else is happening 
    % --> Test 2 might profit from the information around 3.2-3.3 Volts
    % Phase shift information, blurre, but in there! This becomes clear
    % when looking at the standardized data.
    % Regression coefficients standard deviations do not explode in the
    % high voltage.
    % The confidence in this region is pretty much fine! (Even though the
    % SNR might be low, the lines are still sorted ok by cycle life).
    % It really all depends on the X y relationship and how this interplays
    % with the data. Models van be reasonably robust to data with low snr.
end

%% Incorporating SNR --> SNR scaling
if cycle_life_response_snr_db_scaling
    disp("Cycle Life Response SNR DB Scaling:")
    lfp_snr_smooth_dB = readmatrix("data/lfp_snr_smooth_dB.csv", "NumHeaderLines", 0);
    [test, c_X_snr_dB_cl, s_X_snr_dB_cl] = normalize(X_train_);
    figure; 
    plot(x_lfp, test)
    lfp_snr_smooth_dB = (lfp_snr_smooth_dB-min(lfp_snr_smooth_dB)+1e-4)/(max(lfp_snr_smooth_dB)-min(lfp_snr_smooth_dB));
    scale_factor_snr_db = s_X_snr_dB_cl./(lfp_snr_smooth_dB');
    
    X_train_snr_scaled_dB_ = normalize(X_train_, "scale", scale_factor_snr_db);
    X_test1_snr_scaled_dB_ = normalize(X_test1_, "center", c_X_snr_dB_cl, "scale", scale_factor_snr_db);
    X_test2_snr_scaled_dB_ = normalize(X_test2_, "center", c_X_snr_dB_cl, "scale", scale_factor_snr_db);
    
    [b_cl_snr, b0_cl_snr, retval_cl_snr] = bayesreg_wrap(X_train_snr_scaled_dB_, y_train_, false); 
    [mean_b_std, std_b_std] = plot_bayesian_reg_coef(b_cl_snr, retval_cl_snr, x_lfp);

    predict(X_train_snr_scaled_dB_, X_test1_snr_scaled_dB_, X_test2_snr_scaled_dB_, y_train, y_test1, y_test2, c_y_cl, s_y_cl, b_cl_snr, true)
end


%% 2nd Attempt with SNR not in DB! 
if cycle_life_response_snr_scaling
    disp("Cycle Life Response SNR Scaling:")
    lfp_snr_smooth = readmatrix("data/lfp_snr_smooth.csv", "NumHeaderLines", 0);
    [test, c_X_snr_dB_cl, s_X_snr_dB_cl] = normalize(X_train_);
    lfp_snr_smooth = (lfp_snr_smooth-min(lfp_snr_smooth)+1e-4)/(max(lfp_snr_smooth)-min(lfp_snr_smooth));
    scale_factor_snr_db = s_X_snr_dB_cl./(lfp_snr_smooth');
    
    X_train_snr_scaled_ = normalize(X_train_, "scale", scale_factor_snr_db);
    X_test1_snr_scaled_ = normalize(X_test1_, "center", c_X_snr_dB_cl, "scale", scale_factor_snr_db);
    X_test2_snr_scaled_ = normalize(X_test2_, "center", c_X_snr_dB_cl, "scale", scale_factor_snr_db);
    
    [b_cl_snr, b0_cl_snr, retval_cl_snr] = bayesreg_wrap(X_train_snr_scaled_, y_train_, false); 
    [mean_b_std, std_b_std] = plot_bayesian_reg_coef(b_cl_snr, retval_cl_snr, x_lfp);

    predict(X_train_snr_scaled_, X_test1_snr_scaled_, X_test2_snr_scaled_, y_train, y_test1, y_test2, c_y_cl, s_y_cl, b_cl_snr, true)
end


%% Quick check of PLS regression
disp("Cycle Life Response SNR Scaling PLS:")
for i=1:8
    disp(i)
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train_snr_scaled_,y_train_,i);
    predict(X_train_snr_scaled_, X_test1_snr_scaled_, X_test2_snr_scaled_, y_train, y_test1, y_test2, c_y_cl, s_y_cl, BETA(2:end), false)
end

disp("Cycle Life Response std PLS:")  
for i=1:8
    disp(i)
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X_train_std,y_train_,i);
    predict(X_train_std, X_test1_std, X_test2_std, y_train, y_test1, y_test2, c_y_cl, s_y_cl, BETA(2:end), false)
end



%% Finalize the Bayesian Anlaysis, ghenerate data and make figures in python
% for the SI. 

% It should be noted that amrginal improvemets in RMSE might not be
% significant and can be a stupid tradeoff of model complexity.

% Furthermore: Need awareness, that the model is brittle, thus we should be
% honest in reporting that. RMSE of test 2 is highly influenced by how the
% model is trained.

% Check responses of the Bayesian anlysis: Look in the documentation, howe
% to 

%% Functions
function [b, b0, retval] = bayesreg_wrap(X, y, std_bool)
    [b, b0, retval] = bayesreg( ...
        X, ...
        y, ...
        't', ...
        'ridge', ...
        'nsamples',1e4, ...
        'burnin',1e3, ...
        'thin',1, ...
        'display',false, ...
        'normalize', std_bool, ...
        'tau2prior', 'uniform' ...
    );
end

function [mean_b, std_b] = plot_bayesian_reg_coef(b, retval, x_lfp, beta_true)
    % Plot the results
    % Make a figure that resembles the figure in the notebook. 
    figure; 
    mean_b = mean(b,2);
    std_b = std(b, 0, 2);
    plot(x_lfp, mean_b')
    hold on
    plot(x_lfp, mean_b'+std_b')
    plot(x_lfp, mean_b'-std_b')
    hold on
    if exist('beta_true','var')
     % third parameter does not exist, so default it to something
      plot(x_lfp, beta_true)
    end
    figure; 
    plot(x_lfp, retval.tStat)
end

function rmse_val = rmse(y, y_pred)
    rmse_val = sqrt(mean((y-y_pred).^2));
end

function [y_train_pred, y_test1_pred, y_test2_pred] = predict_base(X_train_, X_test1_, X_test2_, mean_y_train, std_y, beta)
    y_train_pred_ = X_train_*beta;
    y_test1_pred_ = X_test1_*beta;
    y_test2_pred_ = X_test2_*beta;

    y_train_pred = exp(((y_train_pred_*std_y)+mean_y_train)); 
    y_test1_pred = exp(((y_test1_pred_*std_y)+mean_y_train)); 
    y_test2_pred = exp(((y_test2_pred_*std_y)+mean_y_train)); 
end

function predict(X_train_, X_test1_, X_test2_, y_train, y_test1, y_test2, mean_y_train, std_y, beta, plot_bool)
    beta_mean = mean(beta,2);
    [y_train_pred, y_test1_pred, y_test2_pred] = predict_base(X_train_, X_test1_, X_test2_, mean_y_train, std_y, beta_mean);
    
    y_train = exp(y_train);
    y_test1 = exp(y_test1); 
    y_test2 = exp(y_test2);

    fprintf('RMSE Train: %.2f \n', rmse(y_train, y_train_pred))
    fprintf('RMSE Test1: %.2f \n', rmse(y_test1, y_test1_pred))
    fprintf('RMSE Test2: %.2f \n', rmse(y_test2, y_test2_pred))

    if plot_bool
        figure; 
        scatter(y_train, y_train_pred, "Marker", "*");
        hold on;
        scatter(y_test1, y_test1_pred, "Marker", "+");
        scatter(y_test2, y_test2_pred, "Marker", ">");
        xlabel("Cycle Life Observed")
        ylabel("Cylce Life Predicted")
        line = linspace(0, 2500, 5);
        plot(line, line)
        axis equal; 
        xlim([0, 2500]);
    end
    
    if size(beta, 2) > 1
        repeats = size(beta,2);
        rmse_vals_train = zeros(repeats, 1);
        rmse_vals_test1 = zeros(repeats, 1);
        rmse_vals_test2 = zeros(repeats, 1);
    
        for i=1:size(beta,2)
            b = beta(:, i);
            [y_train_pred, y_test1_pred, y_test2_pred] = predict_base(X_train_, X_test1_, X_test2_, mean_y_train, std_y, b);
            rmse_vals_train(i) = rmse(y_train, y_train_pred);
            rmse_vals_test1(i) = rmse(y_test1, y_test1_pred);
            rmse_vals_test2(i) = rmse(y_test2, y_test2_pred);
        end
        
        fprintf('Mean RMSE Train: %.2f \n', mean(rmse_vals_train))
        fprintf('Std RMSE Train: %.2f \n', std(rmse_vals_train))
        fprintf('Mean RMSE Test1: %.2f \n', mean(rmse_vals_test1))
        fprintf('Std RMSE Test1: %.2f \n', std(rmse_vals_test1))
        fprintf('Mean RMSE Test2: %.2f \n', mean(rmse_vals_test2))
        fprintf('Std RMSE Test2: %.2f \n', std(rmse_vals_test2))

    end

end