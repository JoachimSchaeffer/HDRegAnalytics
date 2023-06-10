function [Z, prednames, XtoZ, groups, minmaxX] = br_expandvars(X, vars)

[n,px] = size(X);

% Create an empty predictor matrix
Z            = zeros(n, vars.pz);
prednames    = cell(1, vars.pz+1);
XtoZ         = zeros(1, vars.pz);
isvarexp     = false(1, vars.pz);
minmaxX      = NaN(2, vars.pz);

%% Create a predictor matrix from a data matrix
k = 1;
for j = 1:px
    %% If no expansion
    if (~vars.isVarCat(j))
        % Pass data straight through
        if (istable(X))
            Z(:,k) = X{:,j};
        else
            Z(:,k) = X(:,j);
        end
        
        minmaxX(:,j) = minmax(Z(:,k)');
        
        prednames{k} = vars.varnames{j};
        XtoZ(k) = j;            
        k = k+1;

    %% Otherwise, if categorical variable
    else
        if (length(vars.Categories{j}) > 2)
            isvarexp(j) = true;
        end
        
        % Expand into a number of columns equal to number of categories-1
        for i = 2:length(vars.Categories{j})
            if (istable(X))
                Z(X{:,j} == vars.Categories{j}{i}, k) = 1;
                prednames{k} = sprintf('%s.%s', vars.varnames{j}, vars.Categories{j}{i});
            else
                Z(X(:,j) == vars.Categories{j}(i),k) = 1;
                prednames{k} = sprintf('%s.%d', vars.varnames{j}, vars.Categories{j}(i));
            end
           
            XtoZ(k) = j;
            k = k+1;
        end
    end
end

%% Finally, if variables were expanded, all the expanded predictors associated with a variable should be grouped together
if (any(isvarexp))
    groups{1} = nan(1, vars.pz);
    Ix = find(isvarexp);
    
    % Assign predictors to associated variable groups
    g = 1;
    for i = Ix
        groups{1}(XtoZ == i) = g;
        g = g+1;
    end
else
    groups = cell(1,0);
end

%% Done
prednames{end} = '_cons';

return;