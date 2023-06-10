function modelstats = br_compute_model_stats(y, X, retval)
%BR_COMPUTE_MODEL_STATS Compute model fit statistics for model summaries
%
%   The input arguments are:
%       X          - [n x p] data matrix
%       y          - [n x 1] target vector
%       beta       - [p x 1] regression parameters
%       beta0      - [1 x 1] intercept parameter
%       retval     - struct containing sampling information
%
%   Return values:
%       modelstats - [1 x 1] model fit statistics
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2017

%% Model type
gaussian = false;
laplace = false;
tdist = false;
binomial = false;
poisson = false;
geometric = false;
switch retval.runstats.model
    case {'binomial', 'logistic'}
        binomial = true;
        model = 'binomial';
    case {'gaussian', 'normal'}
        gaussian = true;
        model = 'gaussian';
    case {'laplace', 'l1'}
        laplace = true;    
        model = 'laplace';
    case {'t', 'studentt'}
        tdist = true;
        model = 't';
    case 'poisson'
        poisson = true;
        model = 'poisson';
    case 'geometric'
        geometric = true;
        model = 'geometric';
end

%% Stats for continuous model
if (~binomial)
    mu = X*retval.muB + retval.muB0;

    modelstats.logl = -br_regnlike(model, X, y, retval.muB, retval.muB0, retval.muSigma2, retval.runstats.tdof);
    modelstats.r2 = 1 - sum((y - mu).^2) / sum((y - mean(y)).^2);      
    
    % Stats for geometric/Poisson
    if (poisson || geometric)
        % Pseudo R^2
        modelstats.logl0 = -br_regnlike(model, X, y, zeros(retval.Xstats.px, 1), log(mean(y)));
        modelstats.r2    = 1.0 - modelstats.logl / modelstats.logl0;    
        
        % Overdispersion estimate
        if (poisson)
            modelstats.overdispersion = mean( (y-exp(mu)).^2 ./ exp(mu) );
        elseif (geometric)
            geo_p = 1./(exp(mu)+1);
            modelstats.overdispersion = mean( (y-exp(mu)).^2 ./ ((1-geo_p)./geo_p.^2) );
        end
    end

%% Stats for binary model
else
    modelstats.logl  = -br_regnlike(model, X, y, retval.muB, retval.muB0);
    modelstats.logl0 = -br_regnlike(model, X, y, zeros(retval.Xstats.px, 1), log(sum(y)/(retval.Xstats.nx - sum(y))));
    modelstats.r2    = 1.0 - modelstats.logl / modelstats.logl0;    
end

end