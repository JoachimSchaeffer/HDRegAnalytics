function [groups, groupID] = br_build_groups(G, p)

nGroupLevels = 1;
groups = {nan(1, p)};
maxGroup = zeros(1, 1);
nGroups = length(G);
groupID = {zeros(1,p)};

%% Sort all groups
for i = 1:nGroups
    G{i} = sort(G{i});
end

%% Convert list of groups into (multiple levels) of "wide" group indices
for i = 1:nGroups
    %% Error checking to ensure group is specified correctly
    if (min(G{i}) < 1 || max(G{i}) > p)
        error('Group %d indexes variables outside of [1, %d]', i, p);
    end
    if (length(unique(G{i})) ~= length(G{i}))
        error('Group %d indexes the same variable more than once', i);
    end
    if (any(floor(G{i}) ~= G{i}))
        error('Group %d contains variable indices that are not whole numbers', i);
    end
    if (length(G{i}) == 1)
        error('Group %d is a singleton group -- a group must contain between 2 and %d variables', i, p-1);
    end
    if (length(G{i}) == p)
        error('Group %d includes all variables -- a group must contain between 2 and %d variables', i, p-1);
    end
    for k = 1:(i-1)
        if (length(G{i}) == length(G{k}))
            if (all(G{i} == G{k}))
                error('Group %d and %d are identical -- all groups must be different', i, k);
            end
        end
    end
    
    %% Find the first group level in which this group fits without overlaps
    wGroupLevel = nan;
    for j = 1:nGroupLevels
        if (all(isnan(groups{j}(G{i}))))
            wGroupLevel = j;
            break;
        end
    end
    
    %% If it cannot fit into any without overlapping, creating a new group level
    if (isnan(wGroupLevel))
        nGroupLevels = nGroupLevels+1;
        groups{nGroupLevels} = nan(1, p);
        maxGroup(nGroupLevels) = 0;
        wGroupLevel = nGroupLevels;
        groupID{nGroupLevels} = zeros(1,p);
    end
    
    %% Store into the specified group level
    maxGroup(wGroupLevel) = maxGroup(wGroupLevel)+1;
    groups{wGroupLevel}(G{i}) = maxGroup(wGroupLevel);
    
    % Store the original ID in the mapping variable
    groupID{wGroupLevel}(maxGroup(wGroupLevel)) = i;
end

%% Finally, add all variables not in a group in each level to a fake extra group (for convenience)
for j = 1:nGroupLevels
    %groups{j}(isnan(groups{j})) = maxGroup(j)+1;
    groupID{j} = groupID{j}(1:maxGroup(j));
end

return