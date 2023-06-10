%% Example 10
% This example demonstrates how to use the "groups" option in bayesreg.
% We will now recreate br_example9, which used MATLAB tables, but
% this time we will only use MATLAB matrices. In this case,
% we have to explicitely tell bayesreg how to do the grouping
% Grouping of variables works only with HS, HS+ and lasso priors. Note that
% the same variable can appear in multiple groups.
clear;

fprintf('Example 10 - Bayesian logistic regression with groups\n');

rng(1);

load heart;

%% Attribute information for heart data
% Variables 3, 7 and 13 are categorical with more than 2 category
% -- 1. age 
% -- 2. sex 
% -- 3. chest pain type (4 values) 
% -- 4. resting blood pressure 
% -- 5. serum cholestoral in mg/dl 
% -- 6. fasting blood sugar > 120 mg/dl 
% -- 7. resting electrocardiographic results (values 0,1,2) 
% -- 8. maximum heart rate achieved 
% -- 9. exercise induced angina 
% -- 10. oldpeak = ST depression induced by exercise relative to rest 
% -- 11. the slope of the peak exercise ST segment 
% -- 12. number of major vessels (0-3) colored by flourosopy 
% -- 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect 


%% Convert categorical variables to dummy coding
X(:,7) = X(:,7) + 1; % coding is now [1 2 3] instead of [0 1 2]

% convert coding of X(:,13) to [1 2 3] instead of [3 6 7]
ix = X(:,13) == 3;
X(ix,13) = 1;
ix = X(:,13) == 6;
X(ix,13) = 2;
ix = X(:,13) == 7;
X(ix,13) = 3;

chestpain = dummyvar(X(:,3));
recg = dummyvar(X(:,7));
thal = dummyvar(X(:,13));

noncat = [1,2,4:6,8:12];
Z = [X(:, noncat) chestpain(:,2:end) recg(:, 2:end) thal(:,2:end)];

%% Run bayesreg to predict absence (0) or presence (1) of heart disease 
% We have 3 groups, one group for each of the three categorical predictors.
% Columns 11:13 correspond to group 1 (original predictor chest pain type)
% Columns 14:15 correspond to group 2 (original predictor resting ECG)
% Columns 16:17 correspond to group 3 (original predictor thal)
[b, b0, retval] = bayesreg(Z,y,'binomial','hs','nsamples',1e4,'burnin',1e4,'thin',5,'display',true,'groups',{[11 12 13],[14 15],[16 17]});

%% To see how bayesreg groups predictors, type
retval.grouping

%% Compute predictions
fprintf('\n');
fprintf('----------------------------------------------------------------------\n');
[pred, predstats] = br_predict(Z, b, b0, retval, 'CI', [2.5, 97.5], 'ytest', y, 'display', true);
