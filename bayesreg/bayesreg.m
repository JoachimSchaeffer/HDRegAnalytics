function [beta, beta0, retval] = bayesreg(X, y, model, prior, varargin)
%BAYESREG Bayesian linear and logistic regression.
%   [beta, beta0, retval] = bayesreg(...) is the main function of the
%   toolbox and the only one the user is likely to call directly. This
%   function implementes Bayesian linear and logistic regression with
%   continuous shrinkage priors. Once a model is trained, prediction 
%   can be done using br_predict(). Summary statistics for a model 
%   are obtainable using br_summary().
%
%   The input arguments are:
%       X             - [n x p] data matrix (without the intercept); OR
%                       [1 x 1] MATLAB table
%       y             - [n x 1] target vector (if X is a matrix); OR
%                       [1 x 1] string containing name of target variable (if X is a table)
%       model         - string, one of {'gaussian','laplace','t','binomial','poisson','geometric'}
%       prior         - string, one of {'g','ridge','lasso','horseshoe','horseshoe+','logt'}
%       varargin      - optional arguments described below.
%
%   The following optional arguments are supported in the format 'argument', value:
%       'nsamples'    - number of posterior samples (Default: 1000)  
%       'burnin'      - number of burnin samples (Default: 1000)
%       'thin'        - level of thinning (Default: 5)
%       'display'     - do we display summary statistics? (Default: true)
%       'displayor'   - display odds ratios in logreg? See examples\br_example3 (Default: false)
%       'varnames'    - a cell array containing names of predictors. See
%                       examples\br_example3 (Default: {'v1','v2',...'}) 
%       'sortrank'    - display variables in rank order? (Default: false)
%       'tdof'        - degrees of freedom parameter for Student t dist. See
%                       examples\br_example1 (Default: 5) 
%       'catvars'     - vector of variables (as column numbers of X) to treat
%                       as categorical variables, with appropriate expansion. 
%                       See examples\br_example5 (Default: none)
%       'nogrouping'  - stop automatic grouping of categorical predictors
%                       that is enabled with the 'catvars' options. (Default: false)
%       'groups'      - create groups of predictors. Grouping of variables
%                       works only with HS, HS+ and lasso prior
%                       distributions. The same variable can appear in
%                       multiple groups. See examples\br_example[9,10,11,12]  (Default: { [] } )  
%       'tau2prior'   - selects prior for tau2; this can be: (1) a string
%                       'hc' (default),'sb' or 'uniform' corresponding to the
%                       half-Cauchy, Strawderman-Berger or uniform on the
%                       shrinkage parameter; or (2) a vector of length two
%                       specifying the [a,b] hyperparameters of the beta
%                       prior on the shrinage parameter (Default: [0.5,0.5])
%       'blocksample' - sample beta coefficients in blocks (useful for
%                       large p). The parameter specifies how many blocks
%                       to use.  See examples\br_example13.m
%       'blocksize'   - sample beta coefficients in blocks (useful for
%                       large p). The parameter specifies the approximate
%                       size of each block. See examples\br_example13.m
%       'waic'        - whether to calculate the WAIC -- disabling can lead
%                       to large speed-ups, especially for Gaussian models with large n
%                       (default: true)
%
%   Return values:
%       beta        - [p x NSAMPLES] posterior samples of beta
%       beta0       - [1 x NSAMPLES] posterior samples of intercept
%       retval      - additional sampling information (e.g., hyperparameters)
%
%
%   Please see the scripts in the folder "\examples" for usage examples.
%     br_example1   - Fits a univariate Gaussian regression model. Shows
%                     how to use br_predict() to make predictions.
%     br_example2   - Fits a multivariate Gaussian regression model.
%                     Demonstrates how to use br_summary()
%     br_example3   - Fits a logistic regression model with the HS prior.
%     br_example4   - Fits a logistic regression model with the HS+ prior
%                     and two-way interactions in the design matrix.
%     br_example5   - Demonstrates linear regression with MATLAB tables and
%                     the 'catvars' options.
%     br_example6   - Demonstrates logistic gression with MATLAB tables.
%     br_example7   - Demonstrates how to use categorical predictors.
%     br_example8   - Demonstrates linear regression with MATLAB tables and
%                     how to predict using new data.                      
%     br_example9   - Demonstrates grouping of categorical predictors with
%                     the option 'groups'.
%     br_example10  - Demonstrates grouping of categorical predictors with
%                     the option 'groups'.
%     br_example11  - Demonstrates grouping of continuous predictors with
%                     the option 'groups'.
%     br_example12  - Demonstrates grouping with overlapping groups (the
%                     same predictor(s) belong to more than one group).
%     br_example13  - Demonstrates block sampling of betas and how to set
%                     the prior for tau2.
%     br_example14  - Another demonstration of block sampling for moderate
%                     p and larger n
%     br_example15  - How to use br_sparsify() to obtain sparse estimates
%                     of regression coefficients post sampling with bayesreg
%     
%
%   To cite this toolbox:
%     Makalic, E. & Schmidt, D. F.
%     High-Dimensional Bayesian Regularised Regression with the BayesReg Package
%     arXiv:1611.06649 [stat.CO], 2016
%
%   References:
%     Berger, J. O. & Strawderman, W. E. 
%     Choice of Hierarchical Priors: Admissibility in Estimation of Normal Means 
%     The Annals of Statistics, Vol. 24, pp. 931-951, 1996
%
%     Bhadra, A.; Datta, J.; Polson, N. G. & Willard, B. 
%     The Horseshoe+ Estimator of Ultra-Sparse Signals 
%     Bayesian Analysis, 2016
%
%     Bhattacharya, A.; Chakraborty, A. & Mallick, B. K. 
%     Fast sampling with Gaussian scale-mixture priors in high-dimensional regression 
%     arXiv:1506.04778, 2016
%
%     Carvalho, C. M.; Polson, N. G. & Scott, J. G. 
%     The horseshoe estimator for sparse signals 
%     Biometrika, Vol. 97, pp. 465-480, 2010
%
%     Makalic, E. & Schmidt, D. F. 
%     A Simple Sampler for the Horseshoe Estimator 
%     IEEE Signal Processing Letters, Vol. 23, pp. 179-182, 2016
%
%     Schmidt, D. F. & Makalic, E
%     Bayesian Generalized Horseshoe Estimation of Generalized Linear Models
%     Proceedings of the European Conference on Machine Learning (ECML)
%     2019, Wurzburg, Germany
%
%     Schmidt, D. F. & Makalic, E
%     Log-Scale Shrinkage Priors and Adaptive Bayesian Global-Local Shrinkage Estimation
%     arXiv:1801.02321 [math.ST], 2019
%
%     Park, T. & Casella, G. 
%     The Bayesian Lasso 
%     Journal of the American Statistical Association, Vol. 103, pp. 681-686, 2008
%
%     Polson, N. G.; Scott, J. G. & Windle, J. 
%     Bayesian inference for logistic models using Pólya-Gamma latent variables 
%     Journal of the American Statistical Association, Vol. 108, pp. 1339-1349, 2013
%
%     Ray, P. & Bhattacharya, A.
%     Signal Adaptive Variable Selector for the Horseshoe Prior
%     arXiv:1810.09004 [stat.ME], 2018
%
%     Rue, H. 
%     Fast sampling of Gaussian Markov random fields 
%     Journal of the Royal Statistical Society (Series B), Vol. 63, pp. 325-338, 2001
%
%     Xu, Z., Schmidt, D.F., Makalic, E., Qian, G. & Hopper, J.L.
%     Bayesian Grouped Horseshoe Regression with Application to Additive Models
%     AI 2016: Advances in Artificial Intelligence, pp. 229-240, 2016
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016-2020

%% Data dimensions
[nx, px] = size(X);
[ny, py] = size(y);

%% Version number
VERSION = '1.91';

%% Constants
MAX_PRECOMPUTED_PX = 2e4;

%% Parse options
inParser = inputParser;  

%% Default parameter values
defaultNsamples = 1000; 
defaultBurnin = 1000;
defaultThin = 5;
defaultNormalize = true;
defaultDisplay = true;
defaultRunBFR = true;
defaulttDOF = 5;
defaultVarNames = {};
defaultSortRank = false;
defaultDisplayOR = false;
defaultCatVars = [];
defaultNoGrouping = false;
defaultGroups = [];
defaultNumBlocks = [];
defaultBlockSize = [];
defaultTau2Prior = [0.5, 0.5];
defaultWAIC = true;
defaultTruncTau2 = false;
defaultGhs_ab = [1/2, 1/2];
defaultMaxDisplay = inf;

expectedModel= {'gaussian', 'normal', 'laplace', 't', 'studentt', 'binomial', 'logistic', 'poisson', 'gaussianmh', 'binomialmh', 'geometric'};
expectedPrior = {'ridge','rr','horseshoe','hs','lasso','hs+','horseshoe+','gprior','g','logt','loglaplace','ghs','ghorseshoe'};
expectedTau2Prior = {'hc','sb','uniform'};

%% Define parameters

%% Required
addRequired(inParser, 'model', @(x) any(validatestring(x,expectedModel)));
addRequired(inParser, 'prior', @(x) any(validatestring(x,expectedPrior)));

% Optional
addParameter(inParser,'varnames', defaultVarNames, @iscell);
addParameter(inParser,'nsamples',defaultNsamples,@(x) isnumeric(x) && isscalar(x) && (x > 1));
addParameter(inParser,'burnin',defaultBurnin,@(x) isnumeric(x) && isscalar(x) && (x > 0));
addParameter(inParser,'thin',defaultThin,@(x) isnumeric(x) && isscalar(x) && (x > 0));
addParameter(inParser,'normalize',defaultNormalize, @islogical);
addParameter(inParser,'display',defaultDisplay, @islogical);
addParameter(inParser,'rank',defaultRunBFR, @islogical);
addParameter(inParser,'sortrank',defaultSortRank, @islogical);
addParameter(inParser,'displayor',defaultDisplayOR, @islogical);
addParameter(inParser,'tdof',defaulttDOF,@(x) isnumeric(x) && isscalar(x) && (x > 0));
addParameter(inParser,'catvars',defaultCatVars,@(x) isnumeric(x) && all(floor(x) == x) && (min(x) > 0) && max(x) <= px && length(unique(x)) == length(x));
addParameter(inParser,'nogrouping',defaultNoGrouping, @islogical);
addParameter(inParser,'groups',defaultGroups, @(x)(iscell(x) && ~isempty(x)));
addParameter(inParser,'blocksample',defaultNumBlocks, @(x) isnumeric(x) && isscalar(x) && (x > 1) && (x <= px));
addParameter(inParser,'blocksize',defaultBlockSize, @(x) isnumeric(x) && isscalar(x) && (x > 0));
addParameter(inParser,'tau2prior',defaultTau2Prior, @(x) (isnumeric(x) && (length(x) == 2) && (min(x) > 0)) || any(validatestring(x,expectedTau2Prior)) );
addParameter(inParser,'waic',defaultWAIC, @islogical);
addParameter(inParser,'trunctau2', defaultTruncTau2, @islogical);
addParameter(inParser,'ghs_ab', defaultGhs_ab, @(x) (isnumeric(x) && (length(x) == 2) && (min(x)>0)));
addParameter(inParser,'maxdisplay',defaultMaxDisplay, @(x) (isnumeric(x) && (x>0) && (floor(x)==x)));

%% Parse options
parse(inParser, model, prior, varargin{:});  

% General options for all samplers
model = lower(validatestring(model,expectedModel));
prior = lower(validatestring(prior,expectedPrior));
normalize = inParser.Results.normalize; 
nsamples = inParser.Results.nsamples;
burnin = inParser.Results.burnin;
thin = inParser.Results.thin;
display = inParser.Results.display;
sortrank = inParser.Results.sortrank;
runBFR = inParser.Results.rank;
varnames = inParser.Results.varnames;
tdof = inParser.Results.tdof;
displayor = inParser.Results.displayor;
catvars = inParser.Results.catvars;
nogrouping = inParser.Results.nogrouping;
groups_param = inParser.Results.groups;
nBlocks = inParser.Results.blocksample;
approxBlockSize = inParser.Results.blocksize;
tau2prior = inParser.Results.tau2prior;
waic = inParser.Results.waic;
trunctau2 = inParser.Results.trunctau2;
ghs_a = inParser.Results.ghs_ab(1);
ghs_b = inParser.Results.ghs_ab(2);

%% Model type
MH = false;
SMN = false;
gaussian = false;
laplace = false;
tdist = false;
binomial = false;
poisson = false;
geometric = false;
gammadistr = false;
switch model
    case {'binomial', 'logistic'}
        binomial = true;
        model = 'binomial';
        SMN = true;
    case {'gaussian', 'normal'}
        gaussian = true;
        model = 'gaussian';
        SMN = true;
    case {'laplace', 'l1'}
        laplace = true;    
        model = 'laplace';
        SMN = true;
    case {'t', 'studentt'}
        tdist = true;
        model = 't';
        SMN = true;        
    case {'poisson'}
        poisson = true;
        model = 'poisson';
        MH = true;
    case {'geometric'}
        geometric = true;
        model = 'geometric';
        MH = true;
end

%% Prior type
gprior = false;
ridge = false;
lasso = false;
horseshoe = false;
horseshoeplus = false;
logt = false;
loglaplace = false;
ghorseshoe = false;
switch prior
    case {'gprior','g'}
        gprior = true;
        prior = 'g';
        nogrouping = true;
    
    case {'ridge','rr'}
        ridge = true;
        prior = 'ridge';
        nogrouping = true;

    case {'lasso'}
        lasso = true;
        prior = 'lasso';
        
    case {'horseshoe','hs'}
        horseshoe = true;
        prior = 'horseshoe';
        
    case {'horseshoe+','hs+'}
        horseshoeplus = true;
        prior = 'horseshoe+';
        
    case {'logt'}
        alpha = 3;
        logt = true;
        prior = 'logt';
        
    case {'loglaplace'}
        loglaplace = true;
        prior = 'loglaplace';
        
    case {'ghs', 'ghorseshoe'}
        ghorseshoe = true;
        prior = 'ghorseshoe';        
end
logscale = loglaplace | logt;        

%% Type of tau2 prior
if(ischar(tau2prior))    
    switch(lower(validatestring(tau2prior,expectedTau2Prior)))
        case 'hc'
            tau_a = 0.5;
            tau_b = 0.5;
        case 'sb'
            tau_a = 0.5;
            tau_b = 1.0;            
        case 'uniform'
            tau_a = 1.0;
            tau_b = 1.0;            
    end
else
    tau_a = tau2prior(1);
    tau_b = tau2prior(2);
end

%% If input is a table, do some error checking and extract the target
if (istable(X))
    % Some error checking
    if (~ischar(y) || (ischar(y) && ~any(strcmp(X.Properties.VariableNames, y))))
        error('If X is a table, y must contain the name of a variable in the table to use as a target')
    end
    target_var = y;
    
    % Extract the target from the table, and remove it
    y = X{:,target_var};
    target_cats = [];
    if (iscategorical(y))
        target_cats = categories(y);
    end
    X(:,target_var) = [];
    px = px - 1;
    
    % If logistic regression is requested, the target variable must be a
    % binary categorical variable
    if (binomial)
        if (~iscategorical(y) || (iscategorical(y) && length(categories(y)) ~= 2))
            error('For logistic regression target variable must be a binary categorical variable.');
        end
        y = dummyvar(y);
        y(:,1) = [];
    
    % If non-logistic regression, check to ensure target is not a category
    elseif (iscategorical(y))
        error('For continuous regression target variable must not be categorical.');
    end
    
    py = 1;
    ny = length(y);
    
    % Use the table column names as variable names
    varnames = X.Properties.VariableNames;
end
    
%% Check list of predictor names
if((~isempty(varnames)) && (length(varnames) ~= px))
    error('List of variable names is incorrect size');
end

if(~isempty(varnames))
    for i = 1:px
        if(~ischar(varnames{i}))
            error('All variable names must be of type string');
        end
    end
end

% Create temporary variable names if no variable names were passed
if(isempty(varnames))
    for i = 1:px
        varnames{i} = sprintf('v%d', i);
    end
end
varnames{px+1} = '_cons';  


%% Handle any groups that were passed by the user
groups = cell(1,0);
groupID = cell(1,0);
if (~isempty(groups_param))
    if(gprior || ridge)
        error('Grouping variables currently not allowed with ridge and g priors');
    end
    if (nogrouping)
        error('If groups are specified ''nogrouping'' must be false');
    end
    [groups, groupID] = br_build_groups(groups_param, px);
end

%% Setup variable processing rules
vars.description         = 'Variable information';
vars.XTable              = istable(X);
if (istable(X))
    vars.target_var      = target_var;
    if (binomial)
        vars.target_cats = target_cats;
    end
end
vars.varnames            = varnames;
vars.isVarCat            = false(1, px);
vars.isVarCat(catvars)   = true;
vars.Categories          = cell(1, px);
vars.isVarExp            = false(1, px);

%% If the input is a table, ensure categorical variables are to be expanded
vars.pz = px;
if (istable(X))
    % If categorical variables were nominated from command line, report an
    % error
    if (any(vars.isVarCat))
        error('Do not manually specify categorical variables using the ''catvars'' option when you are using a Table');
    end
    
    % Build the predictor matrix, and determine which variables are 
    % categorical and will require categorical expansion
    for i = 1:px
        % If the variable is categorical, we need to expand it
        % appropriately
        if (iscategorical(X{:,i}))
            % Get the categories from the categorical variable, and remove
            % any categories that have no occurences as we cannot learn
            % from these
            vars.Categories{i} = categories(X{:,i});
            if (any(countcats(X{:,i}) == 0))
                warning('Categorical variable ''%s'' defines categories that do not appear in the data.', X.Properties.VariableNames{i});
                vars.Categories{i}(countcats(X{:,i}) == 0) = [];
            end
            
            vars.isVarCat(i)   = true;
            vars.isVarExp(i)   = true;
            
            vars.pz = vars.pz + length(vars.Categories{i}) - 2;
        end
    end

%% Else if the input is not a table, and categorical variables were indicated
else
    for i = find(vars.isVarCat)
        % Get the categories
        vars.Categories{i} = unique(X(:,i));
        
        % Check to ensure all unique elements are non-negative integers
        if (any(unique(X(:,i)) ~= floor(unique(X(:,i)))) || any(unique(X(:,i)) < 0 ))
            error('Categorical variables must contain only non-negative integers; variable ''%s'' violates this condition.', varnames{i});
        end
        
        vars.isVarCat(i) = true;
        vars.isVarExp(i) = true;
        
        vars.pz = vars.pz + length(vars.Categories{i}) - 2;
    end
end

%% Transform input variable matrix into predictor matrix as required
[X, varnames, vars.XtoZ, vars.exp_groups, vars.minmaxX] = br_expandvars(X, vars);
px = vars.pz;

%% If groups were passed by the user, we need to handle any new predictors that arose from expansion appropriately
if (~isempty(groups))
    for j = 1:length(groups)
        groups{j} = groups{j}(vars.XtoZ);
    end
end

%% Add the groups from expansion to any other specified groupings
if (~nogrouping)
    groups(length(groups)+1 : length(groups)+length(vars.exp_groups)) = vars.exp_groups;
end
nGroupLevels = length(groups);

%% Finally, pre-compute number of groups and group sizes
nGroups = zeros(nGroupLevels, 1);
GroupSizes = cell(nGroupLevels, 1);
for j = 1:nGroupLevels
    nGroups(j) = max(groups{j});
    GroupSizes{j} = zeros(nGroups(j), 1);
    for i = 1:nGroups(j)
        GroupSizes{j}(i) = sum(groups{j} == i);
    end
end

%% Error checking
% Check that there are no NaNs in y
if(any(isnan(y)))
    error('Target y contains NaNs');
end

% Check that there are no NaNs in X
if(any(isnan(X(:))))
    error('Covariate matrix X contains NaNs');    
end

% Check dimensions of X and y
if(nx ~= ny)
    error('Dimensions of X and y are incompatible');
end

if(py ~= 1)
    error('Target y must be a vector of size [n x 1]');
end

% If Poisson/geometric regression is requested, the target variable must contain
% non-negative integers
if (poisson || geometric)
    if (any(floor(y) ~= y) || any(y<0))
        error('For count regression target variable must contain only non-negative integers.');
    end
end

% Make sure std(X) > 0 for all columns
colcheck = ~all(std(X));
if(colcheck)
    error('One or more x variables with variance zero detected');
end

%% Check targets
switch model
    case {'gaussian', 'laplace', 't', 'poisson', 'geometric'}
        if(std(y) == 0)
            error('Target y has zero variance');
        end
        z = y;
 
    case {'binomial'}
        u = unique(y);
        if(length(u) ~= 2)
            error('Target y must be coded as {0, 1}');
        else
            if(~all(u == [0;1]))
                error('Target y must be coded as {0, 1}');
            end
        end   
end
weights = laplace | tdist | binomial; 

%% Check that g-prior is usable (X must be full rank)
if(gprior)
    if(nx < px)
        error('Matrix X is not full rank. Cannot use the g-prior.');
    else
        [~,R] = qr(X,0);
        rankx = sum(abs(diag(R)) > abs(R(1))*nx*eps);
        if(rankx < px)
            error('Matrix X is not full rank. Cannot use the g-prior.');
        end
    end
end

%% Normalize data?
if(~normalize)
    muX = zeros(1, px);
    normX = ones(1, px);    

elseif(normalize)
    [X, muX, normX] = standardise(X);
end

%% Block sampling?
blocksample = false;
if(~isempty(nBlocks) && ~isempty(approxBlockSize))
    error('Please specify either ''blocksample'' or ''blocksize'', but not both.');
end
if(~isempty(nBlocks) || ~isempty(approxBlockSize))
    blocksample = true;
end

%% Return values
retval.version    = VERSION;
beta0             = zeros(1, nsamples);
beta              = zeros(px, nsamples);

retval.sparsify_method = '';
retval.sparseB0   = [];
retval.sparseB    = [];
retval.medB0      = [];
retval.medB       = [];
retval.muB0       = 0;
retval.muB        = zeros(px, 1);
retval.tau2       = zeros(1, nsamples);
retval.xi         = zeros(1, nsamples);

if(~binomial)
    retval.sigma2   = zeros(1, nsamples);
    retval.muSigma2 = 0;    
end

if(~(ridge || gprior))
    retval.lambda2 = zeros(px, nsamples);
end

if (nGroupLevels > 0)
    retval.delta2 = cell(1, nGroupLevels);
    for j = 1:nGroupLevels
        retval.delta2{j} = zeros(max(groups{j}), nsamples);
    end
end

%% Initial values
b = randn(px, 1);
sigma2 = 1;
e = [];
tau2 = 1;
xi = 1e-3;
lambda2 = ones(px, 1);
omega2 = ones(nx, 1);
nu = ones(px, 1);
phi2 = ones(px, 1);
zeta = ones(px, 1);
psi2 = 1;
psi2_mu = 0;
w2 = ones(px, 1);
XtX = [];
Xty = [];
Xt1 = [];
negll = zeros(1, nsamples);
waicProb = zeros(nx, 1);
waicLProb = zeros(nx, 1);
waicLProb2 = zeros(nx, 1);

starttime = tic;

%% if blocksampling, determine sizes of each block and start/end locations
blocksize = [];
blockStart = [];
blockEnd = [];
BlockXtX = [];
BlockXty = [];
BlockXty_update = [];

if(blocksample)
    if(~isempty(approxBlockSize))  % is user specified approximate block size
        nBlocks = round(px / approxBlockSize);  % determine # of blocks
    end
   
    % size of each block; start and end coordinates of each block
    blocksize = floor(px / nBlocks) + ((1:nBlocks) <= mod(px,nBlocks));
    blockStart = [1, 1+cumsum(blocksize(1:end-1))]; 
    blockEnd = cumsum(blocksize);
end

%% Determine sampling algorithm for betas
mvnrue = true;            % Use Rue's MVN sampling algorithm
if(px/nx >= 2)
   mvnrue = false;        % else use Bhat. 
end

if(blocksample)             % the block size is now the same as px
    if(blocksize(1)/nx < 2) % use Rue alg unless blocksize is large compared to nx
        mvnrue = true;
    end
end

%% Precompute OK?
precompute = false;
if (gprior && ~blocksample && px >= MAX_PRECOMPUTED_PX)
    error('The g-prior does not currently work on px >= %d without block-sampling');
end
if (((gaussian && mvnrue) || gprior) && px <= MAX_PRECOMPUTED_PX)
    precompute = true;
    if (gaussian)
        yty = z'*z;
        Xty = X'*z;    
    end
    XtX = X'*X;
end

%% Always precompute mean(z) if Gaussian
mu_z = [];
if (gaussian)
    mu_z = mean(z);
    if (~normalize)
        Xt1 = sum(X)';
    end
end

%% Pre-computing for Gaussian block-sampling
if((gaussian && blocksample && mvnrue) || (gprior && blocksample))
    % precompute blocks of X'X if block sampling betas
    BlockXtX = cell(nBlocks, 1); 
    BlockXty = cell(nBlocks, 1);
    
    % If p is small enough we can also compute the additional Xk'*X(-k)
    % blocks for quick updating of X'*y
    if (precompute)
        BlockXty_update = cell(nBlocks, 1);
    end
    for k = 1 : nBlocks
        ix = ((1:px) >= blockStart(k)) & ((1:px) <= blockEnd(k));
        
        % If full precomputed XtX, Xty available use these
        if (precompute)
            BlockXtX{k} = XtX(ix,ix);
            if (gaussian)
                BlockXty{k} = Xty(ix);
                BlockXty_update{k} = XtX(ix,~ix);
            end
            
        % otherwise build them from scratch 
        else
            BlockXtX{k} = X(:,ix)'*X(:,ix);
            if (gaussian)
                BlockXty{k} = X(:,ix)'*z;
            end
        end
    end
end

if(binomial)
    kappa = (y - 0.5);
end

%% Marginal gradient based sampling setup
if (MH)
    mhsampler = 'mgrad1';
    
    % If no normalization, report an error
    if (~normalize)
        error('Metropolis-Hastings based samplers may only be used with normalization on.');
    end
    
    % If insufficient burnin available
    if (burnin < 1e3)
        error('To use Metropolis-Hastings sampling you must use at least 1,000 burnin samples.');
    end
    
    %% Initialise beta's and b0 with rough ridge solutions
    % Poisson
    if (poisson)
        extra_model_params = [];
        Xty = X'*y;        
        [b, b0, ~, ~, ~] = br_FitGLMGD(X, y, model, 1, 10, 500);
        [L_b, grad_b, H_b0, eta, extra_stats] = br_mGradL(y, X, [b;b0], [], model, extra_model_params, Xty);
    % Geometric
    elseif (geometric)
        extra_model_params = 1;
        Xty = X'*y;
        [b, b0, ~, ~, ~] = br_FitGLMGD(X, y, model, 1, 10, 1000);
        [L_b, grad_b, H_b0, eta, extra_stats] = br_mGradL(y, X, [b;b0], [], model, extra_model_params, Xty);
    end
    grad_b = grad_b(1:end-1);
    
    mh_tuning = mh_Initialise(75, 1e2, 1e-7, burnin, false);
        
    % Select desired MH sampler
    switch (lower(mhsampler))
        case 'mgrad1'
            mGrad1 = true;
            
        case 'mgrad2'
            mGrad2 = true;
            invS = X'*X;
    end
end

%% Setup hyperparameters for groups, if required
delta2 = cell(1, nGroupLevels);
rho    = cell(1, nGroupLevels);
if (horseshoeplus)
    rho_a = cell(1, nGroupLevels);
    rho_b = cell(1, nGroupLevels);
end
for j = 1:nGroupLevels
    delta2{j} = ones(max(groups{j})+1, 1);
    rho{j}    = ones(max(groups{j})+1, 1);
    if (horseshoeplus)
        rho_a{j} = ones(max(groups{j})+1, 1);
        rho_b{j} = ones(max(groups{j})+1, 1);
    end
end

%% Statistics for result structure retval
% Run statistics
retval.runstats.description = 'Run arguments';
retval.runstats.model = model;
retval.runstats.prior = prior;
retval.runstats.nsamples = nsamples;
retval.runstats.burnin = burnin;
retval.runstats.thin = thin;
retval.runstats.normalize = normalize;
retval.runstats.rank = runBFR;
retval.runstats.sortrank = sortrank;
retval.runstats.displayor = displayor;
retval.runstats.blocksample = blocksize;
retval.runstats.tdof = tdof;
retval.runstats.tau2prior = [tau_a, tau_b];

if(~tdist)
    retval.runstats.tdof = [];
end

% Store information related to grouping of variables
retval.grouping.nGroupLevels = nGroupLevels;
retval.grouping.groups = groups;
if (length(groupID) < length(groups))
    if (isempty(groupID))
        groupID{1} = 1:max(groups{1});
    else
        groupID{end+1} = (max(groupID{end})+1) : (max(groupID{end}) + max(groups{end}));
    end
end
retval.grouping.groupID = groupID;
retval.grouping.nGroups = 0;
if (~isempty(groupID))
    retval.grouping.nGroups = max(groupID{end});
end
retval.grouping.groupIx = cell(1, retval.grouping.nGroups);
i = 1;
for j = 1:nGroupLevels
    for k = 1:max(groups{j})
        retval.grouping.groupIx{i} = find(groups{j} == k);
        i = i+1;
    end
end

for j = 1:nGroupLevels
    groups{j}(isnan(groups{j})) = max(groups{j})+1;
end

% X statistics
retval.Xstats.description = 'Predictor matrix statistics';
retval.Xstats.varnames = varnames;
retval.Xstats.nx = nx;
retval.Xstats.px = px;
retval.Xstats.muX = muX;
retval.Xstats.normX = normX;

%% Banner
if(display)
    % First, determine the length of the largest variable name
    maxlen = 12;
    for i = 1:px
        if (length(varnames{i}) > maxlen)
            maxlen = length(varnames{i});
        end
    end    
    
    fprintf('%s\n', repchar('=', maxlen + 85));
    fprintf('|%s|\n',centrestr(sprintf('Bayesian Penalised Regression Estimation ver. %s', VERSION), maxlen + 83));
    fprintf('|%s|\n',centrestr(sprintf('(c) Enes Makalic, Daniel F Schmidt. 2017-20'), maxlen+83));
    fprintf('%s\n', repchar('=', maxlen + 85));
end


%% Gibbs sampling
k = 0;
iter = 0;
while(k < nsamples)
    %% Form the diagonal "Lambda" matrix
    [~, delta2prod] = make_Lambda(sigma2, tau2, lambda2, groups, delta2);    
    
    %% If using SMN sampler for beta/beta0
    if (SMN)
        %% Sample beta0
        if(binomial)
            z = kappa .* omega2;
        end
        [b0, muB0] = sample_beta0(X, z, mu_z, Xt1, b, sigma2, omega2);   

        %% Sample beta
        [b, muB] = sample_beta(X, z, mvnrue, b0, sigma2, tau2, lambda2, delta2prod, omega2, XtX, Xty, Xt1, weights, gprior, b, blocksample, blocksize, blockStart, blockEnd, BlockXtX, BlockXty, BlockXty_update);

        % Compute linear predictor if required (gprior, no precomputation, n < p)
        eta = [];
        if (gprior || isempty(XtX) || nx < px || waic)
            eta = X*b + b0;
        end
        
        %% Sample sigma2    
        if(~binomial)
            % Fast computation if n > p and precomputed ...
            ete = [];
            if (isempty(eta))
                if (isempty(Xt1))
                    ete = yty - 2*Xty'*b + b'*XtX*b + b0^2*nx - 2*mu_z*nx*b0;
                else
                    ete = yty - 2*Xty'*b + [b;b0]'*[XtX,Xt1;Xt1',nx]*[b;b0] - 2*mu_z*nx*b0;
                end
            end

            [sigma2, muSigma2, e] = sample_sigma2(eta, y, b, ete, omega2, tau2, lambda2, delta2prod, gprior);
        end        
        
    %% If using MH sampler for beta/beta0
    elseif (MH)
        %% Sample beta
        mh_tuning.D = mh_tuning.D+1;
        if (gammadistr)
            % for future implementation ...
        else
            [b, L_b, grad_b, eta, H_b0, MH_accepted, extra_stats] = br_mGradSampleBeta(b, b0, L_b, grad_b, sigma2*tau2*lambda2.*delta2prod, mh_tuning.delta, eta, H_b0, y, X, sigma2, Xty, model, extra_model_params, extra_stats);
        end
        
        % If it was accepted?
        if (MH_accepted)
            mh_tuning.M = mh_tuning.M+1;
        end

        % Update as required
        muB = b;
        muB0 = b0;        
        
        %% Sample sigma2, if needed
        % Gaussian sigma2
        if (strcmp(model,'gaussian'))
            [sigma2, muSigma2, e] = sample_sigma2(eta, y, b, [], omega2, tau2, lambda2, delta2prod, gprior);
        else
            muSigma2 = 1;
        end
        
        %% Sample b0 -- only every 25 ticks
        if (mod(iter,25) == 0)
            b0_new = normrnd(b0, sqrt(2.5/H_b0));
            switch(model)
                case 'poisson'
                    [L_bnew, grad_bnew, H_b0new, ~, extra_stats_new] = br_mGradL(y, X, [b;b0], eta-b0+b0_new, model, extra_model_params, Xty);
                case 'geometric'
                    [L_bnew, grad_bnew, H_b0new, ~, extra_stats_new] = br_mGradL(y, X, [b;b0], eta-b0+b0_new, model, extra_model_params, Xty);                    
                %case 'gaussian'
                %    [L_bnew, grad_bnew, H_b0new, extra_stats_new] = mGrad_gaussianL_eta(y, X, mu-b0+b0_new);
                %case 'binomial'
                %    [L_bnew, grad_bnew, H_b0new, extra_stats_new] = mGrad_binomialL_eta(y, X, mu-b0+b0_new);
                %case 'gamma'
                %    [L_bnew, grad_bnew, H_b0new, extra_stats_new] = mGrad_gammaL_eta(y, X, mu-b0+b0_new);
            end        

            % Accept?
            if (rand < exp(-L_bnew/sigma2 + L_b/sigma2))
                eta = eta - b0 + b0_new;
                b0 = b0_new;
                L_b = L_bnew;
                H_b0 = H_b0new;
                grad_b = grad_bnew(1:end-1);
                extra_stats = extra_stats_new;
            end        
        end        
    end        

    %% Sample omega2
    if(weights)
        % Logistic regression
        if(binomial)
            omega2 = sample_omega2_logistic(eta);
        
        % Linear regression with Laplace errors
        elseif(laplace)
            omega2 = sample_omega2_laplace(e, sigma2);
        
        % Linear regression with t errors
        elseif(tdist)
            omega2 = sample_omega2_tdist(e, sigma2, tdof);                    
        end        
    end    

    %% Sample tau2
    % HS/HS+/ridge/logscale all use tau2 ~ C+(0,1) => tau2 ~ IG(tau_a, 1/xi), xi ~ IG(1/2, tau_b)
    if (~lasso)
        % Untruncated
        if (~logscale && ~trunctau2)
            [tau2, ~] = sample_tau2(b, sigma2, lambda2, delta2prod, xi, eta, gprior, tau_a);

            %% Sample xi
            xi = sample_xi(tau2, tau_a + tau_b);                
            
        % Truncated
        else
            tau2 = sample_tau2_trunc(b, sigma2, lambda2, delta2prod, eta, gprior, nx);
        end            
        
    % Lasso uses tau2 ~ IG(1, 1)
    else
        tau2 = sample_tau2(b, sigma2, lambda2, delta2prod, 1, [], false, 1);        
    end


    %% Individual shrinkage hyperparameters
    % LASSO prior
    if(lasso)
        lambda2 = sample_lambda2_lasso(b, sigma2, tau2, delta2prod);

    % HS prior
    elseif(horseshoe)
        lambda2 = sample_lambda2_hs(b, sigma2, tau2, nu, delta2prod);       
        nu = sample_nu_hs(lambda2);

    % HS+ prior
    elseif(horseshoeplus)
        % Parameter expanded HS+ sampler
        lambda2 = sample_lambda2_hs(b, sigma2, tau2, nu, delta2prod.*phi2);
        nu = sample_nu_hs(lambda2);

        phi2 = sample_lambda2_hs(b, sigma2, tau2, zeta, delta2prod.*lambda2);
        zeta = sample_nu_hs(phi2);

        lambda2 = lambda2 .* phi2;        

    % Generalized HS prior
    elseif(ghorseshoe)
        m_hs = 1./nu + b.^2./2./tau2./sigma2./delta2prod;

        lambda2 = igamrnd(ghs_b + 1/2, m_hs);
        lambda2 = max(lambda2, 1e-10);
        lambda2 = min(lambda2, 1e10);
        nu = igamrnd(ghs_a + ghs_b, 1 + 1./lambda2);                            

    % Log-scale priors
    elseif(logscale)
        % Sample the lambda2s, given the w2s and scale psi2
        [lambda2, ~] = sample_lambda2_logscale(b, sigma2, tau2, delta2prod, w2, psi2);

        lambda2 = max(lambda2,1e-10);
        lambda2 = min(lambda2,1e10);

        loglambda = log(lambda2)/2;
        
        % Sample the w2s as appropriate
        if (logt)
            w2 = sample_w2_logt(loglambda, psi2, alpha);
        elseif(loglaplace)
            w2 = sample_w2_loglaplace(loglambda, psi2);
        end
        
        % Sample the psi2 hyperparameter
        psi2 = 1./hs_rej_p(sum(loglambda.^2./w2/2), px);
        psi2_mu = psi2_mu + psi2;
    end    
    
    %% Group shrinkage hyperparameters
    if (nGroupLevels > 0)
        for j = 1:nGroupLevels
            % Group-LASSO prior
            if (lasso)
                [delta2{j}, delta2prod] = sample_delta2_lasso(b, sigma2, tau2, lambda2, delta2prod, delta2{j}, groups{j}, nGroups(j), GroupSizes{j});
                
            % HS prior
            elseif (horseshoe)
                [delta2{j}, delta2prod] = sample_delta2_hs(b, sigma2, tau2, lambda2, rho{j}, delta2prod, delta2{j}, groups{j}, nGroups(j), GroupSizes{j});
                rho{j} = sample_nu_hs(delta2{j});
                
            % HS+ prior
            elseif (horseshoeplus)
                [delta2{j}, delta2prod] = sample_delta2_hs(b, sigma2, tau2, lambda2, rho{j}, delta2prod, delta2{j}, groups{j}, nGroups(j), GroupSizes{j});
                rho{j} = sample_nu_hsplus(delta2{j}, rho_a{j});
                rho_a{j} = sample_phi2_hsplus(rho{j}, rho_b{j});
                rho_b{j} = sample_zeta_hsplus(rho_a{j});
            end
        end
    end
    
    %% Do we collect samples?
    iter = iter + 1;
    if(iter > burnin)
        % Thinning
        if(mod(iter,thin) == 0)
            k = k + 1;
            
            %% Store posterior samples
            beta0(k)              = b0;    
            beta(:,k)             = b;
            retval.tau2(k)        = tau2;
            if (logscale)
                retval.psi2(k)    = psi2;
            end
            
            %% Posterior means
            retval.muB  = retval.muB + muB;
            retval.muB0 = retval.muB0 + muB0;            
            
            %% Negative log-likelihood of the model 
            if (~isempty(eta))
                [negll(k), lprob, prob] = br_regnlike_mu(model, eta, e, y, sigma2, tdof);
            else
                negll(k) = (nx/2)*log(2*pi*sigma2) + ete/2/sigma2;
                prob = 0;
                lprob = 0;
            end
            
            %% Sufficient statistics required for WAIC calculation
            waicProb = waicProb + prob;
            waicLProb = waicLProb + lprob;
            waicLProb2 = waicLProb2 + lprob.^2;
            
            if(~binomial)
                retval.sigma2(k) = sigma2;
                retval.muSigma2 = retval.muSigma2 + muSigma2;         
            end            
            if(~(ridge || gprior))
                retval.lambda2(:,k) = lambda2;
            end
            
            if (nGroupLevels > 0)
                for j = 1:nGroupLevels
                    retval.delta2{j}(:,k) = delta2{j}(1:end-1);
                end
            end
        end
        
    %% Else we are in the burnin phase; if we are using MH sampler we need
    % to perform step-size tuning
    elseif (iter <= burnin && MH)
        % Tuning step
        mh_tuning = mh_Tune(mh_tuning);
    end
end

%% If Metropolis-Hasting, acceptance probability check
% if (MH)
%     if (mh_tuning.M/mh_tuning.D < 0.5 || mh_tuning.M/mh_tuning.D > 0.6)
%         fprintf('Warning: acceptance probability (~%.3g) not in 0.5-0.6; please increase number of burn-in iterations.\n', mh_tuning.M/mh_tuning.D);
%     end
%     mh_tuning.M/mh_tuning.D
% end

%% Compute average posterior means
retval.muB = retval.muB / nsamples;
retval.muB0 = retval.muB0 / nsamples;
if(~binomial)
    retval.muSigma2 = retval.muSigma2 / nsamples;
end

%% Other statistics
retval.tStat = retval.muB ./ std(beta,[],2);
retval.varranks = nan(px+1,1);
retval.vars = vars;

%% If requested, compute ranks
if (runBFR)
    retval.varranks = bfr(beta);        
end

%% Compute model fit statistcs
retval.modelstats = br_compute_model_stats(y, X, retval);
retval.modelstats.negll = negll;
if (waic)
    retval.modelstats.waic_dof = sum(waicLProb2/nsamples) - sum((waicLProb/nsamples).^2);
    retval.modelstats.waic = -sum(log(waicProb/nsamples)) + retval.modelstats.waic_dof;
else
    retval.modelstats.waic_dof = Inf;
    retval.modelstats.waic = Inf;
end
if (MH)
    retval.modelstats.probaccept = mh_tuning.M/mh_tuning.D;
end

%% Re-scale coefficients?
if(normalize)
    beta = bsxfun(@rdivide, beta, normX');
    beta0 = beta0 - muX*beta;
    
    retval.muB = retval.muB ./ normX';
    retval.muB0 = retval.muB0 - muX * retval.muB;        
end

%% Posterior median estimates
retval.medB = median(beta, 2);
retval.medB0 = median(beta0);

%% Store runtime data
retval.runstats.rundate = datestr(now, 'dddd, dd mmmm yyyy');
retval.runstats.runtime = toc(starttime);

%% Display results
if(display) 
    % Display summary statistics
    br_summary(beta, beta0, retval);
end

end