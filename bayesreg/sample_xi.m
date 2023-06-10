function xi = sample_xi(tau2, tau_ab)
%SAMPLE_XI samples the hyperparameter xi for all models.
%   xi = sample_xi(...) samples global hyperparameter xi
%   from the conditional posterior distribution. All models.
%
%   The input arguments are:
%       tau2   - [1 x 1] global variance hyperparameter
%
%   Return values:
%       xi     - [1 x 1] sample from the posterior distribution
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%% Update of xi
scale = 1 + 1/tau2;
if(tau_ab == 1.0)
    xi = 1 / exprnd_fast(1/scale);
else
    shape = tau_ab;
    xi = scale / randg(shape,1);    
end

end