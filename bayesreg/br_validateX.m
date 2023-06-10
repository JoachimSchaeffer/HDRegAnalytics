function br_validateX(X, retval)

%% First check if was trained on table and what type input is
if (retval.vars.XTable && ~istable(X))
    error('BayesReg model trained on a table -- predict requires a table as input');
end
if (~retval.vars.XTable && istable(X))
    error('BayesReg model trained on a matrix -- predict requires a matrix as input');
end

%% Now check to see no extra variables are included
if (size(X,2) ~= length(retval.vars.varnames)-1)
    error('Number of variables in testing data different from training data -- cannot predict');
end

%% If data is a table, we can do a lot more structural checks
if (istable(X))
    for j = 1:size(X,2)
        % Check if names match
        if (~strcmp(X.Properties.VariableNames{j}, retval.vars.varnames{j}))
            error('Column %d of input table has incorrect variable name ''%s'' -- should be ''%s''.', j, X.Properties.VariableNames{j}, retval.vars.varnames{j});
        end
        
        % If names match, check that types match
        if (retval.vars.isVarCat(j) && ~iscategorical(X{:,j}))
            error('Variable ''%s'' in input table should be categorical.', retval.vars.varnames{j});
        elseif (~retval.vars.isVarCat(j) && iscategorical(X{:,j}))
            error('Variable ''%s'' in input table should not be categorical.', retval.vars.varnames{j});
        end
        
        % If types match, and are categorical, check categories
        if (retval.vars.isVarCat(j))
            c = unique(X{:,j});
            for i = 1:size(c,1)
                if (~any(strcmp(char(c(i)), retval.vars.Categories{j})))
                    error('Category ''%s'' for variable ''%s'' did not appear in training data -- cannot be used for testing.', c(i,:), retval.vars.varnames{j});
                end
            end
        end
    end

%% Otherwise data is a matrix -- less checks possible
else
    for j = 1:size(X,2)
        % Check categorical variables
        if (retval.vars.isVarCat(j))
            c = unique(X(:,j));
            
            % Check if categories are not positive integers
            if (any(c<0) || any(c~=floor(c)))
                error('Categories for variable ''%s'' are not all positive integers.', retval.vars.varnames{j});
            end
            
            % Otherwise check to see no new categories appear in testing data
            s = setdiff(c, retval.vars.Categories{j});
            if (~isempty(s))
                error('Category ''%d'' for variable ''%s'' did not appear in training data; cannot be used for testing.', s(1), retval.vars.varnames{j});
            end
        end
    end
end

end