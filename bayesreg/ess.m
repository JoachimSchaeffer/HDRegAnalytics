function [ESS, ESSfrac] = ess(x)
%ESS computes effective sample size.
%   [ESS, ESSfrac] = ess(...) computes the effective sample size
%   for an MCMC sampling chain.
%
%   The input arguments are:
%       x       - [NSAMPLES x 1] vector of posterior samples
%
%   Return values:
%       ESS     - [1 x 1] effective sample size
%       ESSfrac - [1 x 1] effective sample size as a fraction
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

n = length(x);
s = min(n - 1, 2000);

g = my_autocorr(x, s);

G = g(2:s-1) + g(3:s);
ix = find(G < 0);

ESS = 0;
ESSfrac = 0;
if(~isempty(ix))
    k = ix(1);

    V = g(1) + 2 * sum(g(2:k));
    ACT = V / g(1);
    ESS = min(n / ACT, n);   
    ESSfrac = ESS / n;
end

end

%% Compute autocorrelations quickly using fast Fourier transform
function acf = my_autocorr(y, order)

% Demean 'y'
y = y-mean(y);

% Forward transform / inverse transform
nFFT = 2^(nextpow2(length(y))+1);
F = fft(y, nFFT);
F = F .* conj(F);
acf = ifft(F);

% Normalize and return
acf = acf(1:(order+1)); 
acf = real(acf);
acf = acf./acf(1); 

end