function [dof, fulldof] = br_dof(wGroups, X, retval)
%BR_DOF compute posterior expected degrees-of-freedom for a model and
%       subgroups of the model predictors. 
%
%   The input arguments are:
%       wGroups     - [vector] list of group ID's for which to compute
%                     degrees-of-freedoms. Can be empty or can be the string 'all' 
%                     to compute DOF for all groups in the model
%       X           - [n x p] data matrix or table that was used by
%                     BayesReg to train the regression model
%       retval      - model structure containing sampling information
%
%
%   Returns value:
%       dof         - [vector] column vector of DOF's for groups specified in wGroups
%       fulldof     - [1 x 1] posterior expected degrees-of-freedom for the
%                     entire model. Note: If this return value is not requested,
%                     the full DOF is not computed which can be
%                     substantially faster if only group DOFs are needed
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2017
%
if (~exist('wGroups','var'))
    wGroups = [];
end

%% Warnings
if (~strcmp(retval.runstats.model, 'gaussian'))
    warning('This function currently only computes exact posterior degrees-of-freedom for Gaussian data models -- for non-Gaussian models it will only be approximate');
end

%% Error checking
if (~isempty(wGroups) && ischar(wGroups) && strcmp(wGroups,'all'))
    wGroups = 1:retval.grouping.nGroups;
end
if (~isempty(wGroups))
    if (~isvector(wGroups) || ~isnumeric(wGroups) || length(unique(wGroups)) ~= length(wGroups) || min(wGroups) < 0 || max(wGroups) > retval.grouping.nGroups || any(floor(wGroups) ~= wGroups))
        error('wGroups argument must be a vector of unique integers from [0, %d] or the string ''all'' for all groups', retval.grouping.nGroups);
    end
end

if (retval.vars.XTable && ~istable(X))
    error('BayesReg model trained on a table -- predict requires a table as input');
end
if (~retval.vars.XTable && istable(X))
    error('BayesReg model trained on a matrix -- predict requires a matrix as input');
end

%% Process the design matrix
if (istable(X))
    X(:,retval.vars.target_var) = [];
end
X = br_expandvars(X, retval.vars);
if (retval.runstats.normalize)
    X = standardise(X);
end

%% Precompute X'*X matrices for all the specified groups and set up the invS matrices
invS = cell(1, length(wGroups));
XtX  = cell(1, length(wGroups));
for j = 1:length(wGroups)
    invS{j} = zeros(length(retval.grouping.groupIx{wGroups(j)}));
    XtX{j}  = X(:,retval.grouping.groupIx{wGroups(j)})' * X(:,retval.grouping.groupIx{wGroups(j)});
end
if (nargout == 2)
    invS_full = zeros(size(X,2));
    XtX_full  = X'*X;
end

% Ridge regression/g-prior regression?
nolambda2 = false;
if (~isfield(retval,'lambda2'))
    nolambda2 = true;
    lambda2 = ones(retval.vars.pz, 1);
end

%full_dof_tst = 0;
%dof_tst = zeros(1, length(wGroups));

%% Compute the DOF's
p = size(X, 2);
delta2 = cell(1, retval.grouping.nGroupLevels);
groups = retval.grouping.groups;
for j = 1:retval.grouping.nGroupLevels
    groups{j}(isnan(groups{j})) = max(groups{j})+1;
end
for i = 1:retval.runstats.nsamples
    for j = 1:retval.grouping.nGroupLevels
        delta2{j} = [retval.delta2{j}(:,i); 1];
    end
    if (~nolambda2)
        lambda2 = retval.lambda2(:,i);
    end
    Lambda = 1./make_Lambda(1, retval.tau2(i), lambda2, groups, delta2);
    Lambda = max(Lambda, 1e-3);
    
    %% Update DOF matrices
    for j = 1:length(wGroups)
        %invS{j} = invS{j} + (XtX{j} + diag(Lambda(retval.grouping.groupIx{wGroups(j)}))) \ eye(size(invS{j},1));
        %dof_tst(j) = dof_tst(j) + trace(XtX{j}*((XtX{j} + diag(Lambda(retval.grouping.groupIx{wGroups(j)}))) \ eye(size(invS{j},1))));
        
        pg = size(invS{j},1);
        L = chol(XtX{j} + diag(Lambda(retval.grouping.groupIx{wGroups(j)})), 'lower');
        invS{j} = invS{j} + L' \ (L \ eye(pg));
    end
    
    if (nargout == 2)
        %invS_full = invS_full + (XtX_full + diag(Lambda)) \ eye(size(X,2));
        %full_dof_tst = full_dof_tst + trace(XtX_full * ((XtX_full + diag(Lambda)) \ eye(size(X,2))));
        
        L = chol(XtX_full + diag(Lambda), 'lower');
        invS_full = invS_full + L' \ (L \ eye(p));

    end
end

%% Finally, calculate the DOFs
dof = zeros(length(wGroups), 1);
for j = 1:length(wGroups)
    %dof(j) = trace(XtX{j} * invS{j}) / retval.runstats.nsamples; 
    dof(j) = (invS{j}(:)' * XtX{j}(:)) / retval.runstats.nsamples; 
end

if (nargout == 2)
    %fulldof = trace(XtX_full * invS_full) / retval.runstats.nsamples;
    fulldof = (invS_full(:)' * XtX_full(:))  / retval.runstats.nsamples; 
end

end