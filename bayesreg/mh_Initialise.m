function tune = mh_Initialise(window, delta_max, delta_min, burnin, display)
%MH_INITIALISE create a tuning structure for adaptive step-size tuning
%   tune = mh_Initialise(...) creates a structure for adaptively
%   updating the step-size for a Metropolis-Hastings sampler
%
%   The input arguments are:
%       window    - [1 x 1] the window over which the acceptance-rate is assessed (recommended value: 50)
%       delta_max - [1 x 1] initial guess for maximum value of step-size (recommended value: 1e2)
%       delta_min - [1 x 1] initial guess for minimum value of step-size (recommended value: 1e-7)
%       burnin    - [1 x 1] the burnin period for the MH sampler
%       display   - [1 x 1] whether to display diagnostic plots
%
%   Return values:
%       tune   - [1 x 1] an initial Metropolis-Hastings tuning structure
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2020

tune = struct;

% Step-size setup
tune.M      = 0;
tune.window = window;

tune.delta_max = delta_max;
tune.delta_min = delta_min;

% Start in phase 1
tune.iter            = 0;
tune.burnin          = burnin;
tune.W_phase         = 1;
tune.W_burnin        = 0;
tune.nburnin_windows = floor(burnin/window);
tune.m_window        = zeros(tune.nburnin_windows, 1);
tune.n_window        = zeros(tune.nburnin_windows, 1);
tune.delta_window    = zeros(tune.nburnin_windows, 1);
tune.display         = display;

tune.phase_cnt       = zeros(1,3);

tune.M               = 0;
tune.D               = 0;

tune.b_tune          = [];

% Start at the maximum delta (phase 1)
tune.delta = delta_max;    

end