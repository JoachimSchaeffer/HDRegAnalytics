function [x, M] = rejsample_hs_p_ab_trunc(m, p, a, b, t0, t1)
%REJSAMPLE_HS_P_TRUNC samples variance hyperparameters from a truncated horseshoe-type hierarchy
%   x = rejsample_hs_p_trunc(...) samples variance hyperparameters
%   from the conditional posterior distribution when the prior is a
%   truncated inverted beta prime (generalized horseshoe) distribution
%
%   The input arguments are:
%       m  - [k x 1] the 'm' arguments
%       p  - [k x 1] the 'p' arguments
%       a  - [1 x 1] the 'a' hyperparameter for GHS
%       b  - [1 x 1] the 'b' hyperparameter for GHS
%       t0 - [1 x 1] log-left-truncation point
%       t1 - [1 x 1] log-right-truncation point
%
%   Return values:
%       x - samples from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2019

if (t0 >= t1)
    error('Truncation points must satisfying -inf <= t0 < t1 <= inf');
end

k = length(m);
x = zeros(k, 1);
M = zeros(k, 1);

for i = 1:k
    [x(i), M(i)] = rejsample(m(i), p, a, b, t0, t1);
end

end

%% Rejection sampler
function [x, M] = rejsample(m, p, a, b, t0, t1)

%% Density: 2*exp(-exp(-2*x)*m)*(exp(-2*x))^(1/2*(-2*b + p))*(1 + exp(2*x))^(-a - b)

%% Mode/2nd-derivative around mode
mode = ((1/2)*(log((2*b + 2*m - p + sqrt(8*m*(2*a + p) + (-2*b - 2*m + p)^2))) - log(2*(2*a + p))));
QQ = exp(-2*mode);
Lm = QQ*m + (p-2*b)*mode + (a+b)*log(1/QQ+1);

%% Figure out points either side of the mode
x0 = mode - 0.85/sqrt(p);
x1 = mode + 1.3/sqrt(p);

%% If we are sampling from the tails ...
if (x1 < t0 || x0 > t1)
    % Left-hand point greater than right-truncation
    if (x0 > t1)
        xp = t1;
    % Right-hand point less than left-truncation
    else
        xp = t0;
    end
    
    % Create envelope
    QQ = exp(-2*xp);
    gp = -2*b + (2*(a+b)/QQ)/(1/QQ+1) - 2*QQ*m + p;
    Lm = QQ*m + (p-2*b)*xp + (a+b)*log(1/QQ+1);

    %% Rejection sample
    M = 1;
    done = false;
    while (~done)
        % Sample from proposal
        x = exprnd_trunc(gp,t0,t1);
        f = gp*(x-xp);
        
        % Accept?
        QQ = exp(-2*x);
        g = QQ*m + (p-2*b)*x + (a+b)*log(1/QQ+1) - Lm;
        if (log(rand) < f-g)
            done=true;
        else
            M=M+1;
        end
    end

%% Use the uni-modal envelope
else
    %% Left-hand segment
    QQ = exp(-2*x0);
    g0 = -2*b + (2*(a+b)/QQ)/(1/QQ+1) - 2*QQ*m + p;
    L0 = QQ*m + (p-2*b)*x0 + (a+b)*log(1/QQ+1) - Lm;
    
    %% Right-hand segment
    QQ = exp(-2*x1);
    g1 = -2*b + (2*(a+b)/QQ)/(1/QQ+1) - 2*QQ*m + p;
    L1 = QQ*m + (p-2*b)*x1 + (a+b)*log(1/QQ+1) - Lm;

    %% Meeting points for the three segments
    left = -(L0-g0*x0-0)/g0;
    right = -(L1-g1*x1-0)/g1;

    %% Normalizing constants for the three densities
    % Truncate
    if (left < t0)
        left = t0;
    end
    if (right > t1)
        right = t1;
    end
    if (left > t1)
        left = t1;
    end
    if (right < t0)
        right = t0;
    end

    left_K = exp(-L0+g0*x0) * (exp(-t0*g0) - exp(-left*g0)) / g0;
    right_K = exp(-L1+g1*x1) * (exp(-right*g1) - exp(-t1*g1)) / g1;
    mid_K = (right-left);
    K = left_K+right_K+mid_K;

    %% Sample
    done=false;
    M=1;
    while(~done)
        u = rand;
        if (u < left_K/K)
            x = exprnd_trunc(g0,t0,left);
            f = L0+g0*(x-x0);
        elseif (u < (left_K+mid_K)/K)
            x = rand*(right-left) + left;
            f = 0;
        else
            x = exprnd_trunc(g1,right,t1);
            f = L1+g1*(x-x1);
        end    

        % Accept?
        QQ = exp(-2*x);
        g = QQ*m + (p-2*b)*x + (a+b)*log(1/QQ+1) - Lm;
        if (log(rand) < f-g)
            done=true;
        else
            M=M+1;
        end
    end
end

x = exp(2*x);

end

%% Truncated exponential sampler
function x = exprnd_trunc(alpha,a,b)

p = rand;

%% If either of the end-points is infinite
if (isinf(a) || isinf(b))
    x = -log(1-p)/alpha;
    if (isinf(a))
        x = x+b;
    else
        x = x+a;
    end
    
%% Otherwise if the truncation interval is finite
else
    t = b-a;
    p = rand;

    %% Use log-sum for numerical stability if required
    exp_alpha_t = exp(alpha*t);
    if (isinf(exp_alpha_t))
        v1 = alpha*t + log((1-p));
        v2 = log(p);
        
        mv = v1;
        if (v2 > v1)
            mv = v2;
        end
        x = (alpha*t - (mv + log(exp(v1-mv)+exp(v2-mv))))/alpha;
    else
        x = (alpha*t - log(exp_alpha_t + p - exp_alpha_t*p))/alpha;
    end

    x = x+a;
end
    
end