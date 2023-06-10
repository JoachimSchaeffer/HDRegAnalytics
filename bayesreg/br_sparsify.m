function retval = br_sparsify(X, y, beta, beta0, retval, varargin)
% SPARSIFY POSTERIOR ESTIMATES.
%   br_sparsify(...) calculates sparse estimates of regression coefficients 
%   following posterior MCMC sampling.
%
%   The input arguments are:
%       X           - [n x p] data matrix or table
%       beta        - [p x N] regression parameters
%       beta0       - [1 x N] intercept parameter
%       retval      - struct containing sampling information
%       varargin    - optional arguments described below.
%
%   The following optional arguments are supported in the format 'argument', value:
%       'method'    - sparsification method to use (default = 'ci')
%                   - Signal Adaptive Variable Selector ('savs')
%                   - k-means clustering ('kmeans') 
%                   - use the top-k ranked variables {'topk'}
%       'cisize'    - which credible interval to use for sparsification? (default = [2.5 97.5])      
%       'topk'      - how many variables to retain with the top-k method (default = 10)
%
%   Returns value:
%       retval      - sampling information and sparse beta estimate
%
%   References:
%
%     Bhattacharya, A.; Pati, D.; Pillai, N. S. & Dunson, D. B.
%     Dirichlet-Laplace priors for optimal shrinkage 
%     Journal of the American Statistical Association, Vol. 110, No. 512, pp. 1479-1490, 2015
%
%     Pallavi Ray and Anirban Bhattacharya
%     Signal Adaptive Variable Selector for the Horseshoe Prior
%     arXiv:1810.09004 [stat.ME], 2018
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%% Parse options
inParser = inputParser;  

%% Default parameter values
defaultMethod = 'ci';
defaultCIsize = [2.5 97.5]; 
defaultDisplay = false;
defaultTopK = 10;
expectedMethod = {'savs','ci','kmeans','topk'};

%% Define parameters
addParameter(inParser, 'method', defaultMethod, @(x)any(validatestring(x,expectedMethod)));
addParameter(inParser, 'cisize', defaultCIsize, @(x)isnumeric(x) && min(x) >= 0 && max(x) <= 100 && (length(x) == 2));
addParameter(inParser, 'display', defaultDisplay, @islogical);
addParameter(inParser, 'topk', defaultTopK, @(x)isnumeric(x) && min(x) >= 1);

%% Parse options
parse(inParser, varargin{:});

method  = lower(validatestring(inParser.Results.method,expectedMethod));
CIsize  = sort(inParser.Results.cisize);
display = inParser.Results.display;
topk    = inParser.Results.topk;

%% Deal with tables
X = deal_with_tables(X, retval);

%% Determine whether to use posterior mean or median
est = retval.muB;
est0 = retval.muB0;
if( strcmp(retval.runstats.model, 'binomial') )
    est = retval.medB;
    est0 = retval.medB0;
end

%% Implementation of sparsification method
switch method
    % Sparsify based on the credible interval
    case {'ci'}
        b = ci_sparsify(beta, est, CIsize);
    
    % Signal Adaptive Variable Selector 
    case {'savs'}        
        b = savs_sparsify(X, est);
        
    % Selection using k-keans
    case {'kmeans'}
        b = kmeans_sparsify(beta, retval);
        
    % Top-K
    case {'topk'}
        b = topk_sparsify(retval, topk);
end

%% Save estimates
if(strcmp(method,'ci')) % nicer for printing
    method = 'CI';
end
retval.sparsify_method = method;

%% Re-adjust intercept
retval.sparseB = b;
[badj, b0_adj] = br_FitGLMGD(X*b, y, retval.runstats.model, [], inf, 1e4);

retval.sparseB0 = b0_adj;
retval.sparseB  = retval.sparseB*badj;

if(display)  
    % hack to make br_summary work with br_sparsify
    tempRet = retval;
    tempRet.muB = b;
    tempRet.medB = b;
    br_summary(beta, beta0, tempRet);
end

end

% Sparsify using the k-means approach of Bhattacharya et al. 2015
function b = kmeans_sparsify(beta, retval)

babs = abs(beta);

%% cluster |beta|-s into two clusters, signal and noise 
nsamples = size(beta,2);
M = zeros(nsamples,1);
for i = 1:nsamples
    [idx, ctr] = kmeans(babs(:,i), 2);
    [~, II] = max(ctr);
    M(i) = sum(idx==II);
end

%% set noise coefficients to zero
nonz = mode(M);
b = retval.medB;
[~,ix]=sort(abs(b),'descend');
b(ix(nonz+1:end)) = 0;

end

% Sparsify based on the CI
function b = ci_sparsify(beta, est, CIsize)

b = est; % Start with posterior mean
ci = prctile(beta', CIsize);   % compute credible interval
ix = (ci(1,:) < 0) & (ci(2,:) > 0); % set estimate to zero if CI contains 0
b(ix) = 0;

end

% SAVS 
function b = savs_sparsify(X, est)

% setup
b = est;
babs = abs(b);
Xnorm = sum(X.^2, 1)';
mu = 1 ./ b.^2;

% which regressors to set to zero?
ix = (babs .* Xnorm) <= mu;
b(ix) = 0;
b(~ix) = b(~ix) .* max(0, 1 - mu(~ix) ./  Xnorm(~ix) ./ babs(~ix));

end


% Top-K
function b = topk_sparsify(retval, topk)

[~,I] = sort(retval.varranks(1:end-1));
b = retval.muB;
b(I(topk+1:end)) = 0;

end

function X = deal_with_tables(X, retval)

if (retval.vars.XTable)       
    if (~istable(X))
        error('BayesReg model trained on a table -- br_sparsify requires a table as input');
    end    
    
    % Check to see if the 'y' variable is in the table
    I = find(strcmp(X.Properties.VariableNames, retval.vars.target_var));
    if (~isempty(I))
        X(:,retval.vars.target_var) = [];
    end      
end

%% Now check test 'X' data against the 'X' data model was trained on
br_validateX(X, retval);

%% Handle input data as appropriate
% If input is a table, do some error checking and extract the target
if (istable(X))
    X = br_expandvars(X, retval.vars);

% Else, if data is a matrix, we need a bit of a hack
else
    % HACK -- check if size of 'X' matches expanded version; if so, no
    % expansion needed and we are good -- otherwise expand
    px = retval.Xstats.px;
    if (size(X,2) ~= px)
        X = br_expandvars(X, retval.vars);
    end
end

end