function [pred, predstats] = br_predict(X, beta, beta0, retval, varargin)
%BR_PREDICT produce predictions
%   br_predict(...) provides predictions for a design matrix using a
%   fitted model 
%
%   The input arguments are:
%       X           - [n x p] data matrix or table
%       beta        - [p x 1] regression parameters
%       beta0       - [1 x 1] intercept parameter
%       retval      - struct containing sampling information
%       varargin    - optional arguments described below.
%
%   The following optional arguments are supported in the format 'argument', value:
%       'ytest'     - test data used to evaluate quality of predictions -- use this if
%                     your X is not a table. If X is a table, br_predict will use the 
%                     the appropriate target column of your table for evaluation if it is present
%       'CI'        - percentiles of credible intervals to compute (Default: none)
%       'predictor' - type of predictor to use. Options are 'mean' for
%                     plug-in predictor at the posterior mean, 'median' for
%                     plug-in predictor at the posterior median and
%                     'bayesavg' for a posterior predictive density. (Default: 'mean')
%                     use sparse beta estimates ('sparse')
%
%   Returns value:
%       pred        - [1 x 1] table containing predictions
%       predstats   - [1 x 1] structure containg goodness-of-fit statistics
%                           for predictions if testing data was specified
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

px = retval.Xstats.px;

model = retval.runstats.model;
prior = retval.runstats.prior;
tdof  = retval.runstats.tdof;

%% Model type
gaussian = false;
laplace = false;
tdist = false;
binomial = false;
poisson = false;
geometric = false;
switch model
    case {'binomial', 'logistic'}
        binomial = true;
        model = 'binomial';
    case {'gaussian', 'normal'}
        gaussian = true;
        model = 'gaussian';
    case {'laplace', 'l1'}
        laplace = true;    
        model = 'laplace';
    case {'t', 'studentt'}
        tdist = true;
        model = 't';
    case {'poisson'}
        poisson = true;
        model = 'poisson';
    case {'geometric'}
        geometric = true;
        model = 'geometric';
end

nsamples = size(beta,2);

%% Parse options
inParser = inputParser;  
 
%% Default parameter values
defaultCI = []; 
defaultytest = [];
defaultPredictor = 'mean';
defaultDisplay = false;

expectedPredictor = {'mean','median','bayesavg','sparse'};

%% Define parameters
addParameter(inParser, 'CI', defaultCI, @(x)isnumeric(x) && min(x) >= 0 && max(x) <= 100 && length(x) == 2);
addParameter(inParser, 'ytest', defaultytest, @(x)size(x,2) == 1 && size(x,1) == size(X,1));
addParameter(inParser, 'predictor', defaultPredictor, @(x)any(validatestring(x,expectedPredictor)));
addParameter(inParser, 'display', defaultDisplay, @islogical);

%% Parse options
parse(inParser, varargin{:});  

predictor = lower(validatestring(inParser.Results.predictor,expectedPredictor));
ytest     = inParser.Results.ytest;
CI        = sort(inParser.Results.CI);
display   = inParser.Results.display;

if (strcmp(predictor,'bayesavg'))
    bayesavg = true;
else
    bayesavg = false;
end

useBsparse = false;
if (strcmp(predictor,'sparse'))
    if(isempty(retval.sparseB))
        error('Sparse estimates of regression coefficients not found');
    end
    useBsparse = true;
end

%% Some initial checking
if (retval.vars.XTable)
    if (~istable(X))
        error('BayesReg model trained on a table -- br_predict requires a table as input');
    end
    
    % Check to see if the 'y' variable is in the table -- if so, remove it
    % and store it as testing data
    I = find(strcmp(X.Properties.VariableNames, retval.vars.target_var));
    if (~isempty(I))
        %% If testing data was passed via varargin
        if (~isempty(ytest))
            error('Do not use the ''ytest'' option to specify testing data if your y variable appears in the input table');
        end
        
        ytest = X{:,retval.vars.target_var};
        
        if (binomial)
            if (~iscategorical(ytest) || (iscategorical(ytest) && length(categories(ytest)) ~= 2))
                error('For logistic regression target variable must be a binary categorical variable.');
            end
            ytest = dummyvar(ytest);
            ytest(:,1) = [];

        % If non-logistic regression, check to ensure target is not a category
        elseif (iscategorical(ytest))
            error('For continuous regression target variable must not be categorical.');
        end
        
        X(:,retval.vars.target_var) = [];
    end  

%% 
elseif (~isempty(ytest))
    if (binomial)
        c = sort(unique(ytest));
        if (length(c) ~= 2 || ~all(c==[0;1]))
            error('For logistic regression target variable must be binary with values 0 and 1');
        end
    end
end

%% Now check test 'X' data against the 'X' data model was trained on
br_validateX(X, retval);

nx = size(X,1);

%% Handle input data as appropriate
% If input is a table, do some error checking and extract the target
if (istable(X))
    X = br_expandvars(X, retval.vars);

% Else, if data is a matrix, we need a bit of a hack
else
    % HACK -- check if size of 'X' matches expanded version; if so, no
    % expansion needed and we are good -- otherwise expand
    if (size(X,2) ~= px)
        X = br_expandvars(X, retval.vars);
    end
end

% If no testing data passed, just some dummy testing data
if (isempty(ytest))
    ytest_passed = false;
    ytest = ones(nx, 1);
else
    ytest_passed = true;
end

%% Logistic regression
if (binomial)
    %% If bayesian averaging was not requested
    if (~bayesavg)
        % Probability of success
        if (strcmp(predictor,'mean'))
            [~, ~, prob_y, mu] = br_regnlike(model, X, ytest, retval.muB, retval.muB0, [], []);
        elseif (strcmp(predictor,'median'))
            [~, ~, prob_y, mu] = br_regnlike(model, X, ytest, retval.medB, retval.medB0, [], []);
        elseif(useBsparse)
            [~, ~, prob_y, mu] = br_regnlike(model, X, ytest, retval.sparseB, retval.sparseB0, [], []);                
        end
        prob_1 = prob_y;
        prob_1(ytest == 0) = 1 - prob_1(ytest == 0);
        
    % If Bayesian average was requested
    else
        % Get Bayesian averages of probabilities and log-odds
        mu = zeros(nx, 1);
        prob_y = zeros(nx, 1);
        for j = 1:nsamples
            [~, ~, prob_y_j, mu_j] = br_regnlike(model, X, ytest, beta(:,j), beta0(j), [], []);
            prob_y = prob_y + prob_y_j;
            mu = mu + mu_j;
        end
        
        prob_y = prob_y/nsamples;
        mu = mu/nsamples;
        
        % Get probabilities of success
        prob_1 = prob_y;
        prob_1(ytest == 0) = 1 - prob_1(ytest == 0);
    end

    % Best guess at target class
    if (~retval.vars.XTable)
        yhat = zeros(length(mu), 1);
        yhat(prob_1 > 1/2) = 1;
    else
        yhat = discretize(prob_1, [0, 1/2, 1], 'categorical', retval.vars.target_cats);
    end    

    % Store results
    pred = table(prob_1, yhat, mu);
    pred.Properties.VariableNames{3} = 'logodds';
    
%% Else continuous regression
else
    % Select requested predictor
    % Posterior mean (or bayesian average)
    if (strcmp(predictor,'mean') || bayesavg)
        % Posterior predictions of yhat more stably estimated from
        % posterior mean of beta's (as they are Rao-Blackwellized)
        beta_hat  = retval.muB;
        beta0_hat = retval.muB0;
        
    % Posterior co-ordinatewise median
    elseif (strcmp(predictor,'median'))
        beta_hat  = retval.medB;
        beta0_hat = retval.medB0;
        
    % Posterior estimates sparsified
    elseif (useBsparse)
        beta_hat = retval.sparseB;
        beta0_hat = retval.sparseB0;
        
    end
    eta = X*beta_hat + beta0_hat;
    
    % If Bayesian averaging requested and test data has been passed
    if (bayesavg && ytest_passed)
        % Calculate posterior predictive density 
        prob_y = zeros(nx, 1);        
        mu_avg = zeros(nx, 1);
        for j = 1:nsamples
            [~, ~, prob_y_j, ~, mu_j] = br_regnlike(model, X, ytest, beta(:,j), beta0(j), retval.sigma2(j), tdof);
            prob_y = prob_y + prob_y_j;
            mu_avg = mu_avg + mu_j;
        end
        prob_y = prob_y/nsamples;
        
        %% If Poisson/geometric
        mu = mu_avg/nsamples;
    
    % Else if no averaging requested but testing data passed
    else
        [~, ~, prob_y, ~, mu] = br_regnlike(model, X, ytest, beta_hat, beta0_hat, retval.muSigma2, tdof);
    end
    
    pred = array2table(mu, 'VariableNames', {'yhat'});
end

%% Produce confidence intervals if requested
if (~isempty(CI))
    k = size(pred,2);
    
    % If continuous 
    if (~binomial)
        [~, ~, ~, ~, yhat] = br_regnlike(model, X, ytest, beta, beta0, retval.sigma2, tdof);
        yhat = prctile(yhat', CI)'; 
        
        pred{:,k + 1 : k + length(CI)} = yhat;
        for j = 1:length(CI)
            pred.Properties.VariableNames{k + j} = manglename(sprintf('yhat_CI%f',CI(j)));
        end
        
    % If binary
    else        
        [~, ~, prob_1_CI, ~] = br_regnlike(model, X, ones(nx, 1), beta, beta0, [], []);
        prob_1_CI = prctile(prob_1_CI', CI)';
        
        pred{:,k + 1 : k + length(CI)} = prob_1_CI;
        for j = 1:length(CI)
            pred.Properties.VariableNames{k + j} = manglename(sprintf('prob_1_CI%f',CI(j)));
        end
    end
end

%% If test data was passed, compute probabilites of the test points
if (ytest_passed)
    %% Compute statistics for continuous data
    if (~binomial)
        % Currently: 
        % (i) negative log-likelihood
        % (ii) mean-squared prediction error
        % (iii) mean-absolute prediction error
        % (iv) R^2 value
        %
        %[predstats.neglike, ~, prob_y, mu] = br_regnlike(model, X, ytest, retval.muB, retval.muB0, retval.muSigma2, retval.runstats.tdof);
        predstats.neglike = -sum(log(prob_y));
        predstats.mspe = mean( (mu - ytest).^2 );
        predstats.mape = mean( abs(mu - ytest) );
        if (~poisson && ~geometric)
            predstats.r2 = 1 - predstats.mspe / mean((ytest - mean(ytest)).^2);
        end

    %% Compute statistics for binary data
    else
        % Currently:
        % (i) negative log-likelihood
        % (ii) confusion matrix
        % (iii) classification accuracy
        % (iv) AUC
        %
        %[predstats.neglike, ~, prob_y] = br_regnlike(model, X, ytest, retval.muB, retval.muB0);
        predstats.neglike = -sum(log(prob_y));
        predstats.cm = confusionmat(ytest,double(pred{:,'prob_1'}>1/2));
        predstats.classacc = (predstats.cm(1,1) + predstats.cm(2,2)) / length(ytest);
        [~,~,~,predstats.auc] = perfcurve(ytest, pred{:,'prob_1'}, 1);
        
        P = sum(ytest);
        N = sum(ytest==0);
        TN = predstats.cm(1,1);
        TP = predstats.cm(2,2);
        FP = predstats.cm(1,2);
        FN = predstats.cm(2,1);
        
        predstats.sensitivity = TP / P;
        predstats.specificity = TN / N;
        predstats.ppv = TP / (TP + FP);
        predstats.npv = TN / (TN + FN);
        predstats.F1 = 2*TP / (2*TP + FP + FN);
    end
    
    pred{:, 'prob_y'} = prob_y;
    
    %% If display was requested
    if (display)
        %fprintf('----------------------------------------------------------------------\n');
        if (binomial)
            modeltxt = 'logistic';
        elseif (gaussian)
            modeltxt = 'Gaussian';
        elseif (laplace)
            modeltxt = 'Laplace';
        elseif (tdist)
            modeltxt = 'Student-t';
        elseif (poisson)
            modeltxt = 'Poisson';
        elseif (geometric)
            modeltxt = 'geometric';
        end
        str = ['Bayesian ', modeltxt, ' ', prior, ' regression prediction stats'];
        fprintf('%s\n\n',str);
        
        %fprintf('\n');
        fprintf('%s = ', 'Predictor type                ');
        if (strcmp(predictor,'bayesavg'))
            fprintf('Bayesian posterior predictive density\n');
        elseif (strcmp(predictor,'mean'))
            fprintf('Plug-in (posterior mean)\n');
        elseif (strcmp(predictor,'median'))
            fprintf('Plug-in (posterior median)\n');
        elseif(useBsparse)
            fprintf('Plug-in (sparse, %s)\n', retval.sparsify_method);
        end
        fprintf('\n');        
        
        fprintf('Number of obs                  = %d\n', length(ytest));
        fprintf('%s = %.2f\n', 'Negative log-likelihood       ', predstats.neglike);
        
        % Binomial stats
        if (binomial)
            fprintf('%s = %.3f\n', 'Area-under-the-curve          ', predstats.auc);
            fprintf('%s = %.3f%%\n', 'Classification Accuracy       ', predstats.classacc*100);
                        
            fprintf('\n');
            
            % Confusion matrix
            if (retval.vars.XTable)
                cats = retval.vars.target_cats;
                if (length(cats{1}) > length(cats{2}))
                    max_cat_len = length(cats{1});
                else
                    max_cat_len = length(cats{2});
                end
            else
                cats = {'0', '1'};
                max_cat_len = 1;
            end
                
            cm_str = num2str(predstats.cm(:));
            if (size(cm_str,2) > max_cat_len)
                max_cat_len = size(cm_str,2);
            end
            max_cat_len = max(max_cat_len, 4);
            
            s = sprintf('%%%ds', max_cat_len);
            fprintf('                                 '); 
            fprintf(s,''); fprintf('  ');  fprintf(s, cats{1}); fprintf('  '); fprintf(s, cats{2}); fprintf('\n');
            
            fprintf('Confusion matrix (y, yhat)     = '); fprintf(s, cats{1}); fprintf('  ');
            fprintf(s,cm_str(1,:)); fprintf('  ');
            fprintf(s,cm_str(3,:)); fprintf('\n');
            
            fprintf('                                 '); 
            fprintf(s, cats{2}); fprintf('  ');
            fprintf(s,cm_str(2,:)); fprintf('  ');
            fprintf(s,cm_str(4,:)); fprintf('\n');
            
            %% More statistics
            fprintf('\n');
            fprintf('%s = %.3f\n', 'Sensitivity                   ', predstats.sensitivity);
            fprintf('%s = %.3f\n', 'Specificity                   ', predstats.specificity);
            fprintf('%s = %.3f\n', 'Positive predictive value     ', predstats.ppv);
            fprintf('%s = %.3f\n', 'Negative predictive value     ', predstats.npv);
            fprintf('%s = %.3f\n', 'F1 score                      ', predstats.F1);
            
        else
            fprintf('Mean squared prediction error  = %.2f\n', predstats.mspe);
            fprintf('Mean absolute prediction error = %.2f\n', predstats.mape);
            if (~geometric && ~poisson)
                fprintf('R-squared coefficient          = %.2f\n', predstats.r2);
            end
                     
        end
        %fprintf('----------------------------------------------------------------------\n');
        
    end
    
%% Otherwise
else
    predstats = [];
end

end

%%
function s = manglename(s)

s(s=='.') = '_';
j = length(s);
while (s(j) == '0')
    j = j-1;
end
s = s(1:j);

end