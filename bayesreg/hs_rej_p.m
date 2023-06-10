function [x,M] = hs_rej_p(m,p)

%% Mode/2nd-derivative around mode
mode = (1/2)*(log(sqrt(4*m^2+4*m*(p+3)+(p-1)^2) + 2*m - p + 1) - log(2*(p+1)));
QQ = exp(-2*mode);
Lm = QQ*m + (p-1)*mode + log(1/QQ+1);

%% Left-hand segment
x0 = mode - 0.85/sqrt(p);

QQ = exp(-2*x0);
g0 = -2*QQ*m + (2/QQ)/(1/QQ+1) + p - 1;
L0 = QQ*m + (p-1)*x0 + log(1/QQ+1) - Lm;

%% Right-hand segment
x1 = mode + 1.3/sqrt(p);

QQ = exp(-2*x1);
g1 = -2*QQ*m + (2/QQ)/(1/QQ+1) + p - 1;
L1 = QQ*m + (p-1)*x1 + log(1/QQ+1) - Lm;

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
        x = -(L0+log(-g0*v*left_K)-g0*x0)/g0;
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
    g = QQ*m + (p-1)*x + log(1/QQ+1) - Lm;
    if (log(rand) < f-g)
        done=true;
    else
        M=M+1;
    end
end

x = (1./exp(x))^2;

end