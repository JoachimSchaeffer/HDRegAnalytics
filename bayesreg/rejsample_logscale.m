function loglambda = rejsample_logscale(m, w)

p = length(m);
loglambda = zeros(p, 1);

for i = 1:p
    loglambda(i) = rejsample(m(i), w(i));
end

end

%% Draw a sample using rejection sampling
function [x,M] = rejsample(m, w)

w = min(w,300);

%% Mode/2nd-derivative around mode
mode = (1/2)*(Lambert_W_fast(4*exp(2*w)*m*w) - 2*w);
QQ = exp(-2*mode);
Lm = mode^2/2/w + m*QQ + mode;
H = 4*QQ*m + 1/w;

%% Left-hand segment
x0 = mode - 0.8/sqrt(H);

QQ = exp(-2*x0);
g0 = x0/w - 2*m*QQ + 1;
L0 = x0^2/2/w + m*QQ + x0 - Lm;

%% Right-hand segment
x1 = mode + 1.1/sqrt(H);

QQ = exp(-2*x1);
g1 = x1/w - 2*m*QQ + 1;
L1 = x1^2/2/w + m*QQ + x1 - Lm;

%% Meeting points for the three segments
left = -(L0-g0*x0-0)/g0;
right = -(L1-g1*x1-0)/g1;

%% Normalizing constants for the three densities
left_K=-exp(-L0-g0*(left-x0))/g0;
right_K=exp(-L1-g1*(right-x1))/g1;
mid_K=exp(-0) * (right-left);
K = left_K+right_K+mid_K;

%% Sample
done=false;
M=1;
while(~done)
    u = rand;
    if (u < left_K/K)
        v = rand;
        x = -log(1-v)/g0 + left;
        f = L0+g0*(x-x0);
    elseif (u < (left_K+mid_K)/K)
        x = rand*(right-left) + left;
        f = 0;
    else
        v = rand;
        x = -log(1-v)/g1 + right;
        f = L1+g1*(x-x1);
    end    
    
    % Accept?
    QQ = exp(-2*x);
    g = x^2/2/w + m*QQ + x - Lm;
    if (log(rand) < f-g)
        done=true;
    else
        M=M+1;
    end
end

end

function [w, t] = Lambert_W_fast(x)
% Lambert_W  Functional inverse of x = w*exp(w).
% w = Lambert_W(x), same as Lambert_W(x,0)
% w = Lambert_W(x,0)  Primary or upper branch, W_0(x)
% w = Lambert_W(x,-1)  Lower branch, W_{-1}(x)
%
% See: http://blogs.mathworks.com/cleve/2013/09/02/the-lambert-w-function/

% Copyright 2013 The MathWorks, Inc.

% Effective starting guess
%if nargin < 2 || branch ~= -1
%   w = ones(size(x));  % Start above -1
%else  
%   w = -2*ones(size(x));  % Start below -1
%end
w = 1;
if (x >= 3)
    w = log(x) - log(log(x));
end
v = inf*w;

% Haley's method
t = 0;
while any(abs(w - v)./abs(w) > 1.e-8)
    t = t + 1;
    
    v = w;
    e = exp(w);
    f = w.*e - x;  % Iterate to make this quantity zero
    w = w - f./(e.*(w+1) - (w+2).*f./(2*w+2));
end

end