function br_summary(beta, beta0, retval)
%BR_SUMMARY print summary statistics.
%   summary(...) prints sampling summary statistics in a nicely 
%   formatted table.
%
%   The input arguments are:
%       beta    - [p x 1] regression parameters
%       beta0   - [1 x 1] intercept parameter
%       retval  - struct containing sampling information
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016

%% Extract run parameters
varnames = retval.Xstats.varnames;
nx = retval.Xstats.nx;
px = retval.Xstats.px;

model = retval.runstats.model;
prior = retval.runstats.prior;
nsamples = retval.runstats.nsamples;
burnin = retval.runstats.burnin;
thin = retval.runstats.thin;
normalize = retval.runstats.normalize;
runBFR = retval.runstats.rank;
sortrank = retval.runstats.sortrank;
displayor = retval.runstats.displayor;
tdof = retval.runstats.tdof;
isVarCat = [ retval.vars.isVarCat 0];
XtoZ = [retval.vars.XtoZ length(isVarCat)];

%% Model type
gaussian = false;
laplace = false;
tdist = false;
binomial = false;
poisson = false;
geometric = false;
switch model
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
    case {'poisson'}
        poisson = true;
        model = 'poisson';        
    case {'geometric'}
        geometric = true;
        model = 'geometric';
end

%% Compute ESS for each variable
ESSfrac = zeros(px, 1);
for j = 1:px
    [~, ESSfrac(j)] = ess(beta(j,:));
end    

%% Table symbols
chline = '-';
cvline = '|';
cTT    = '+';

%% Find length of longest variable name
maxlen = 12;
for i = 1:px
    if (length(varnames{i}) > maxlen)
        maxlen = length(varnames{i});
    end
end
fmtstr = sprintf('%%%ds', maxlen);    

%% Display pre-table information
if(binomial)
    modeltxt = 'logistic';
elseif (gaussian)
    modeltxt = 'Gaussian';
elseif (laplace)
    modeltxt = 'Laplace';
elseif (poisson)
    modeltxt = 'Poisson';
elseif (geometric)
    modeltxt = 'geometric';
elseif (tdist)
    modeltxt = sprintf('Student-t (DOF = %.1f)',retval.runstats.tdof);
end

priortxt = prior;
if (strcmp(prior, 'logt'))
    priortxt = 'log-t';
elseif (strcmp(prior, 'loglaplace'))
    priortxt = 'log-Laplace';
end

excess_s = repchar(' ', maxlen - 12);

str = ['Bayesian ', modeltxt, ' ', priortxt, ' regression'];
fprintf('%-71s%sNumber of obs   = %8.0f\n', str, excess_s, nx);
fprintf('%-71s%sNumber of vars  = %8.0f\n', '', excess_s, px);    

% Gaussian/Laplace/Student-t display
if(~binomial && ~geometric && ~poisson)
    s2 = retval.muSigma2;
    if(tdist)
        if (tdof > 2)
            s2 = tdof / (tdof - 2) * s2;
        else
            s2 = nan;
        end
    end
    str = sprintf('MCMC Samples   = %6.0f', nsamples);
    if (~isnan(s2))
        fprintf('%-71s%sstd(Error)      = %8.5g\n', str, excess_s, sqrt(s2));
    else
        fprintf('%-71s%sstd(Error)      =        -\n', str, excess_s);
    end
    str = sprintf('MCMC Burnin    = %6.0f', burnin);
    fprintf('%-71s%sR-squared       = %8.4f\n', str, excess_s, retval.modelstats.r2);    
    str = sprintf('MCMC Thinning  = %6.0f', thin);
    if (~isinf(retval.modelstats.waic))
        fprintf('%-71s%sWAIC            = %8.5g\n', str, excess_s, retval.modelstats.waic);    
    else
        fprintf('%-71s%sWAIC            = %8s\n', str, excess_s, '.');    
    end

% Poisson/geometric display
elseif(geometric || poisson)
    str = sprintf('MCMC Samples   = %6.0f', nsamples);
    
    fprintf('%-71s%sOverdispersion  = %8.5g\n', str, excess_s, retval.modelstats.overdispersion);

    str = sprintf('MCMC Burnin    = %6.0f', burnin);
    fprintf('%-71s%sPseudo R2       = %8.4f\n', str, excess_s, retval.modelstats.r2);    
    str = sprintf('MCMC Thinning  = %6.0f', thin);
    if (~isinf(retval.modelstats.waic))
        fprintf('%-71s%sWAIC            = %8.5g\n', str, excess_s, retval.modelstats.waic);    
    else
        fprintf('%-71s%sWAIC            = %8s\n', str, excess_s, '.');    
    end
    
% Binomial display
elseif(binomial)
    str = sprintf('MCMC Samples   = %6.0f', nsamples);
    fprintf('%-71s%sLog. Likelihood = %8.5g\n', str, excess_s, retval.modelstats.logl);
    str = sprintf('MCMC Burnin    = %6.0f', burnin);
    fprintf('%-71s%sPseudo R2       = %8.4f\n', str, excess_s, retval.modelstats.r2);
    str = sprintf('MCMC Thinning  = %6.0f', thin);
    if (~isinf(retval.modelstats.waic))
        fprintf('%-71s%sWAIC            = %8.5g\n', str, excess_s, retval.modelstats.waic);    
    else
        fprintf('%-71s%sWAIC            = %8s\n', str, excess_s, '.');    
    end
end
fprintf('\n')    

%% Table Header
fprintf('%s%c%s\n', repchar(chline, maxlen+1), cTT, repchar(chline, 83));
tmpstr = sprintf(fmtstr, 'Parameter');
if(binomial && displayor)
    fprintf('%s %c %12s %12s        [95%% Cred. Interval] %10s %7s %9s\n', tmpstr, cvline, 'median(OR)', 'std(OR)', 'tStat', 'Rank', 'ESS');
else        
    fprintf('%s %c %12s %12s        [95%% Cred. Interval] %10s %7s %9s\n', tmpstr, cvline, 'mean(Coef)', 'std(Coef)', 'tStat', 'Rank', 'ESS');        
end
fprintf('%s%c%s\n', repchar(chline, maxlen+1), cTT, repchar(chline, 83));  

%% Variable information
if(runBFR && sortrank)
    [~,indices] = sort(retval.varranks);
else
    indices = 1:px+1;
end

incat = -1;
for i = 1:(px+1)
    k = indices(i);

    % Regression variable
    if (k <= px)
        kappa = retval.tStat(k);             
        s = beta(k,:);
        mu = retval.muB(k);
        if (binomial)
            mu = retval.medB(k);
        end

    % Intercept
    elseif (k == (px+1))
        s = beta0;
        mu = mean(s);
        if(binomial)
            mu = retval.medB0;
        end
    end

    %% Compute credible intervals/standard errors for beta's
    std_err = std(s);
    qlin = prctile(s,[2.5,25,75,97.5]);            
    qlog = prctile(exp(s),[2.5,25,75,97.5]);
    q = qlin;              
    if (binomial && displayor)
        mu = exp(mu);            
        std_err = (qlog(end)-qlog(1))/2/1.96; 
        q = qlog;
    end

    %% Display results
    if(isVarCat(XtoZ(k)))
        if(incat == XtoZ(k)) % if we are printing category values
            % use erase to strip out main category name
            tmpstr = erase([varnames{k}, ' '], [retval.vars.varnames{XtoZ(k)},'.']);
            tmpstr = sprintf(fmtstr, tmpstr);
        else % print the name of the category followed by the first label
            if(i ~= 1)
                blankline = sprintf(fmtstr, ''); 
                fprintf('%s %c\n', blankline, cvline);   
            end
            tmpstr = sprintf(fmtstr, retval.vars.varnames{XtoZ(k)});
            fprintf('%s %c\n', tmpstr, cvline);   
            tmpstr = erase([varnames{k}, ' '], [retval.vars.varnames{XtoZ(k)},'.']);
            tmpstr = sprintf(fmtstr, tmpstr);
            incat = XtoZ(k);
        end
    else
        if(isVarCat(XtoZ(max(1,k-1)))) % if the previous variable was categorical, print a new line
            blankline = sprintf(fmtstr, ''); 
            fprintf('%s %c\n', blankline, cvline);             
        end
        tmpstr = sprintf(fmtstr, varnames{k});
    end
    if(k > px) 
        tstat = '         .';
    else
        tstat = sprintf('%10.3f', kappa);
    end
    if(isnan(retval.varranks(k)))
        rank = '      .';
    else
        rank = sprintf('%7d', retval.varranks(k));
    end
    fprintf('%s %c %s %s   %s %s %s %s', tmpstr, cvline, displaynum(mu,12,5), displaynum(std_err,12,5), displaynum(q(1),12,5), displaynum(q(4),12,5), tstat, rank);

    fprintf(' ');

    %% Display tstat, ESS frac and other stuff
    if ( k <= px && ( (qlin(2) > 0 && qlin(3) > 0) || (qlin(2) < 0 && qlin(3) < 0) ) )    
        fprintf('*');
    else 
        fprintf(' ');
    end
    if ( k <= px && ( (qlin(1) > 0 && qlin(4) > 0) || (qlin(1) < 0 && qlin(4) < 0) ) )    
        fprintf('*');
    else
        fprintf(' ');
    end

    if(k > px)
        fprintf('%7s', '.');
    else
        fprintf('%7.1f', ESSfrac(k)*100);
    end
    fprintf('\n');    

end

%% Bottom line
fprintf('%s%c%s\n', repchar(chline, maxlen+1), cTT, repchar(chline, 83));                 

%% Print elapsed time
endtime = retval.runstats.runtime;
days = floor(endtime / 86400);
endtime = endtime - days * 86400;
hours = floor(endtime / 3600);
endtime = endtime - hours * 3600;
mins = floor(endtime / 60);
secs = endtime - mins * 60;

if(days > 0)
    timestring = sprintf('%d day(s) %d hour(s) %d minute(s) %05.3f seconds', days, hours, mins, secs);
elseif(hours > 0)
    timestring = sprintf('%d hour(s) %d minute(s) %05.3f seconds', hours, mins, secs);
elseif(mins > 0)
    timestring = sprintf('%d minute(s) %05.3f seconds', mins, secs);    
else
    timestring = sprintf('%05.3f seconds', secs);    
end
    
fprintf('Time taken for sampling was %s.\n', timestring);

return

function s = displaynum(x, w, prec)

if (x == 0)
    s = repchar(' ', w);
    s(w - ceil(prec/2)) = '-';
else
    % First we try normal notation
    s = sprintf(sprintf('%%%d.%df', w, prec), x);
    if (length(s) > w)
        s = sprintf(sprintf('%%%d.%dg', w, prec), x);
    end
end

return