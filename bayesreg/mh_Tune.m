function tune = mh_Tune(tune)
%MH_TUNE update tuning parameters for a Metropolis-Hastings sampler
%   tune = mh_Tune(...) updates the adaptive step-size for a 
%   Metropolis-Hastings sampler
%
%   The input arguments are:
%       tune   - [1 x 1] a Metropolis-Hastings tuning structure
%
%   Return values:
%       tune   - [1 x 1] updated Metropolis-Hastings tuning structure
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2020

tune.iter = tune.iter + 1;

DELTA_MAX = exp(40);
DELTA_MIN = exp(-40);
        
%% Perform a tuning step, if necessary
if (mod(tune.iter,tune.window) == 0)
    %W_burnin, W_phase, delta, delta_min, delta_max, mc, nc, delta_window, m_window, n_window)

    % Store the measured acceptance probability
    tune.W_burnin = tune.W_burnin + 1;
    tune.delta_window(tune.W_burnin) = log(tune.delta);
    tune.m_window(tune.W_burnin) = tune.M+1;
    tune.n_window(tune.W_burnin) = tune.D+2;

    %% If in phase 1, we are finding delta_max
    if (tune.W_phase == 1)
        tune.phase_cnt(1) = tune.phase_cnt(1)+1;
        d = linspace(log(DELTA_MIN),log(DELTA_MAX),15);

        %tune.delta = exp(randn / randn);
        %tune.delta = exp(rand*(log(DELTA_MAX) - log(DELTA_MIN)) + log(DELTA_MIN));
        
        tune.delta = exp(d(tune.phase_cnt(1)));
        
        if (tune.delta < tune.delta_min)
            tune.delta_min = tune.delta;
        end
        if (tune.delta > tune.delta_max)
            tune.delta_max = tune.delta;
        end

        if (tune.phase_cnt(1) == 15)
            tune.W_phase = 3;
        end
        
%         % If the probe point does not result in zero acceptance, 
%         % try a larger delta_max
%         if (tune.M ~= 0 && tune.delta < DELTA_MAX)
%             tune.delta = tune.delta*10;
%             tune.delta = min(tune.delta, DELTA_MAX);
%         % Otherwise record this as maximum, and move to phase 2
%         else
%             tune.delta_max = tune.delta;
% 
%             if (tune.delta_min > tune.delta_max)
%                 tune.delta = tune.delta_max/4;
%             else
%                 tune.delta = tune.delta_min;
%             end
% 
%             % Move to phase 2 (searching for delta_min)
%             tune.W_phase = 2;
%         end

    %% If in phase 2, we are finding delta_min
    elseif (tune.W_phase == 2)        
        tune.phase_cnt(2) = tune.phase_cnt(2)+1;
        
        % If the probe point does not result in complete acceptance, 
        % try a smaller delta_min
        if (tune.M ~= tune.D)
            tune.delta = tune.delta/10;
        % Otherwise record this as minimum, and move to phase 3
        else
            tune.delta_min = tune.delta;
            tune.W_phase = 3;

            % Move in between delta_min and delta_max
            tune.delta = sqrt(tune.delta_min*tune.delta_max);
        end

    %% Else in phase 3, we are probing randomly guided by model
    else
        tune.phase_cnt(3) = tune.phase_cnt(3)+1;
        
        %% Fit a logistic regression to the response and generate new random probe point
        tune.b_tune = glmfit(tune.delta_window(1:tune.W_burnin), [tune.m_window(1:tune.W_burnin), tune.n_window(1:tune.W_burnin)], 'binomial');
        probe_p = rand*0.7 + 0.15;
        tune.delta = exp( -(log(1/probe_p-1)+tune.b_tune(1))/tune.b_tune(2) );
        tune.delta = min(tune.delta, DELTA_MAX);
        tune.delta = max(tune.delta, DELTA_MIN);

        if (tune.delta > tune.delta_max)
            tune.delta_max = tune.delta;
        elseif (tune.delta < tune.delta_min)
            tune.delta_min = tune.delta;
        end

        if (tune.delta == tune.delta_max || tune.delta == tune.delta_min)
            tune.delta = exp(rand*(log(tune.delta_max) - log(tune.delta_min)) + log(tune.delta_min));
        end
    end    
    
    %
    tune.M = 0;
    tune.D = 0;
end        

%% If we have reached last sample of burn-in, select a suitable delta
if (tune.iter == tune.burnin)
    % If the algorithm has not grown and shrunk delta, give an error
    if (tune.phase_cnt(3) < 100)
        error('Metropolis-Hastings sampler has not explored the step-size space sufficiently; please increase the number of burnin samples');
    end
  
    tune.b_tune = glmfit(tune.delta_window(1:tune.W_burnin), [tune.m_window(1:tune.W_burnin), tune.n_window(1:tune.W_burnin)], 'binomial');
    
    % Select the final delta to use
    tune.delta = exp( -(log(1/0.55-1)+tune.b_tune(1))/tune.b_tune(2) );
    %if (tune.delta == 0 || isinf(tune.delta))
    %    tune.delta = log(tune.delta_max)/2;
    %end
    
    if (tune.display)
        % Produce a diagonostic plot
        figure(1);
        clf;
        plot(tune.delta_window, tune.m_window./tune.n_window, '.');
        hold on;
        de = linspace(log(tune.delta_min)-5, log(tune.delta_max)+5);
        plot(de, 1./(1+exp(-(de*tune.b_tune(2) + tune.b_tune(1)))), 'LineWidth', 1.5);
        grid on;
        xlabel('$\log(\delta)$','Interpreter','Latex');
        ylabel('Estimated Acceptance Probability','Interpreter','Latex');
        xlim([log(tune.delta_min), log(tune.delta_max)]);

        title(sprintf('mGrad Burnin Tuning: Final $\\delta=%.3g$', tune.delta),'Interpreter','Latex');
    end
end

end