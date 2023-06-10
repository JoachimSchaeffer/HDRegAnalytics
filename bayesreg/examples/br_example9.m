%% Example 9
% This example demonstrates how to group variables in bayesreg.
% In this example, we use MATLAB tables. No special bayesreg commands are
% needed for groups when dealing with categorical n-ary variables (n > 2)
% stored in MATLAB tables. bayesreg automatically expands such categorical
% variables using dummy coding and treats all the expanded columns  for as
% that variable as one group. 
% Grouping of variables works only with HS, HS+ and lasso priors. Note that
% the same variable can appear in multiple groups.
clear;

fprintf('Example 9 - Bayesian logistic regression with groups\n');

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


%% Make a MATLAB table from the data set
noncat = [1,2,4:6,8:12];
T = array2table(X(:,noncat),'VariableNames',varnames(noncat));

% convert categorical data to MATLAB categorical type
chestpain = categorical(num2cell(num2str(X(:,3))), {'1','2','3','4'});
recg = categorical(num2cell(num2str(X(:,7))), {'0','1','2'});
thal = categorical(num2cell(num2str(X(:,13))), {'3','6','7'}, {'normal','fixed defect','reversable defect'});

T{:,'CPAIN'} = chestpain;
T{:,'RECG'} = recg;
T{:,'THAL'} = thal;

% add target to table
heart = categorical(num2cell(num2str(y)), {'0','1'}, {'absent','present'});
T{:,'HEART'} = heart;

%% Run bayesreg to predict absence (0) or presence (1) of heart disease 
%  bayesreg automatically groups all columns in the table associated with a
%  categorical predictor into one group. In this example, we have 3 
%  categorical predictors which means bayesreg creates 3 distinct groups.
%  All other predictors are not grouped.
% 
[b, b0, retval] = bayesreg(T,'HEART','binomial','hs','nsamples',1e4,'burnin',1e4,'thin',5,'display',true);

%% Alternatively, instead of creating Z manually, we could run bayesreg with 'catvars' options
[b, b0, ~] = bayesreg(X,y,'binomial','hs','nsamples',1e4,'burnin',1e4,'thin',5,'varnames',varnames,'display',true,'catvars',[3,7,13]);


%% To see how bayesreg groups predictors, type
retval.grouping

% We see that there are 3 groups, with group IDs 1, 2 and 3.
% Group 1 predictors are at position [11 12 13] in the design matrix and
% correspond to the variable chest pain (4 categories)
% Group 2 columns are at position [14 15] in the design matrix and
% correspond to the variable resting electrocardiographic results (values 0,1,2)

%% We now re-run bayesreg, but this time turn off automatic grouping for
%  categorical predictors using the "nogrouping" option.
%  Notice how the standard errors for (e.g.,) RECG.1/2 have now increased.
[b, b0, retval] = bayesreg(T,'HEART','binomial','hs','nsamples',1e4,'burnin',1e4,'thin',5,'display',true,'nogrouping',true);



