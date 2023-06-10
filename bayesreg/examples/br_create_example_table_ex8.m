%BR_CREATE_EXAMPLE_TABLE Create a data set stored in a MATLAB table.
%
%   Create an example table with a mix of continuous and categorical input 
%   variables (and continuous and binary targets)
%
%   The input arguments are:
%       n       - [1 x 1] sample size
%
%   Return values:
%       T       - [n x 8] Matlab table data (continuous outcome y)
%       Tb      - [n x 8] Matlab table data (binary outcome y)
%       M       - [n x 7] Matrix version of the data (no outcome)
%       y       - [n x 1] Outcome (continuous)
%       yb      - [n x 1] Outcome (binary)
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2017
%
%% 
function [T, Tb, M, y, yb] = br_create_example_table_ex8(n)

%% Setup some data with numerical and categorical variables
X = normrnd(0,1,n,6);
X(:,4) = normrnd(50,10,n,1);
c = categorical(num2cell(num2str(randi(3,n,1))), {'1','2','3'}, {'no','yes','unsure'});

%% Make the table
T = array2table(X(:,1:3),'VariableNames',{'Height','Weight','BP'});
T{:,'AgeGroup'} = discretize(X(:,4), [-inf,40,50,60,70,80,90,inf], 'categorical');
T{:,'S1'} = X(:,5);
T{:,'S2'} = X(:,6);
T{:,'FamHist'} = c;
Tb = T;

% Create data
dv = dummyvar(T{:,'FamHist'});
Z = [X, dv(:,2:end)];

b = [3,-1.5,0,0,0.6,1,2,4]';
y = normrnd(Z*b, sqrt(0.1));
T{:,'Diabetes'} = y;

%% Create a different version with binary target (high, low diabetes status)
Tb{:,'Diabetes'} = discretize(T{:,'Diabetes'},[-inf,median(T{:,'Diabetes'}),inf],'categorical',{'Low','High'});

% Add some random noise to the outcome
for i = 1:n
    if (rand < 0.2)
        if (Tb{i,'Diabetes'} == 'Low')
            Tb{i,'Diabetes'} = {'High'};
        else
            Tb{i,'Diabetes'} = {'Low'};
        end
    end
end

yb = zeros(n, 1);
yb(Tb{:,'Diabetes'} == 'High') = 1;

%% Also create matrix versions of the data
[nT, pT] = size(T);
M = zeros(nT, pT);
for j = 1:pT
    M(:,j) = double(T{:,j});
end

% Remove the outcome variable
M(:,end) = [];

end