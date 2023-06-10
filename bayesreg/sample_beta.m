function [b, muBeta] = sample_beta(X, z, mvnrue, b0, sigma2, tau2, lambda2, delta2prod, omega2, XtX, Xty, Xt1, weights, gprior, b, blocksample, blocksize, blockStart, blockEnd, BlockXtX, BlockXty, BlockXty_update)
%SAMPLE_BETA samples the regression parameters beta.
%   [b, muBeta] = sample_beta(...) samples the regression parameters
%   beta from the conditional posterior distribution.
%
%
%   The input arguments are:
%       X               - [n x p] data matrix 
%       z               - [n x 1] target vector
%       mvnrue          - use Rue's algorithm? {true | false}
%       b0              - [1 x 1] intercept parameter
%       sigma2          - [1 x 1] noise variance
%       tau2            - [1 x 1] global variance hyperparameter
%       lambda2         - [p x 1] local variance hyperparameters
%       delta2prod      - [p x 1] combined group variance hyperparameters
%       omega2          - [n x 1] hyperparameters
%       XtX             - [p x p] pre-computed X'*X (if available)
%       Xty             - [p x 1] pre-computed X'*y (if available)
%       Xt1             - [p x 1] pre-computed X'*1 (if available, only
%                                 required if X not normalized)
%       weights         - [1 x 1] boolean, does the data have weights (i.e. non Gaussian)
%       gprior          - [1 x 1] true for gprior, otherwise false
%       b               - [p x 1] a sample from the posterior distribution
%       blocksample     - [1 x 1] do we sample beta in blocks?
%       blocksize       - [k x 1] size of each beta block
%       blockstart      - [k x 1] start coordinate of each block
%       blockend        - [k x 1] end coordinate of each block
%       BlockXtX        - [1 x 1] cell array of pre-computed X'*X for
%                                 block-sampling, if available
%       BlockXty        - [1 x 1] cell array of pre-computed X'*y for
%                                 block-sampling, if available
%       BlockXty_update - [1 x 1] cell array of pre-computed sub X'*X
%                                 matrices to fast-update X'*y for block-sampling
%
%   Return values:
%       b               - [p x 1] a sample from the posterior distribution
%       muBeta          - [p x 1] posterior mean
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2016-2018

sigma = sqrt(sigma2);
Lambda = sigma2 * tau2 * lambda2 .* delta2prod;

if(~blocksample)
    %% Able to use pre-computed Xty?
    if (weights || isempty(Xty) || ~mvnrue)
        alpha = (z - b0);
    end
    
    % Use Rue's algorithm
    if(mvnrue)
        % Non-Gaussian regression/Gaussian regression without pre-computed XtX/Xty
        if(isempty(Xty) || weights)
            omega = sqrt(omega2);
            X = bsxfun(@rdivide, X, omega);          
            [b, muBeta] = fastmvg_rue(X, [], alpha, [], Lambda, sigma2, omega, gprior, XtX);

        % Gaussian regression with pre-computed XtX/Xty
        else
            if (isempty(Xt1))
                [b, muBeta] = fastmvg_rue([], XtX, [], Xty, Lambda, sigma2, [], gprior, XtX);
            else
                [b, muBeta] = fastmvg_rue([], XtX, [], Xty - b0*Xt1, Lambda, sigma2, [], gprior, XtX);
            end
        end

    % Use Bhat. algorithm
    else
        omega = sqrt(omega2);
        X = bsxfun(@rdivide, X, omega);          
        [b, muBeta] = fastmvg_bhat(X, alpha, Lambda, sigma, omega);
    end
    
    
else
    %% Block sampling
    p = length(lambda2);
    muBeta = zeros(p,1);
    nBlocks = length(blocksize);
    
    
    %% Use Rue's algorithm
    if(mvnrue)
        if (isempty(BlockXty_update) || weights)
            alpha = (z - b0 - X*b);    
        end
        
        %% Non-Gaussian regression
        if(weights)
            Z = [];
            omega = sqrt(omega2);
            for k = 1 : nBlocks
                ix = ((1:p) >= blockStart(k)) & ((1:p) <= blockEnd(k)); % current block to sample
                if(gprior)
                    Z = BlockXtX{k};
                end                
                
                Xscaled = bsxfun(@rdivide, X(:,ix), omega);
                alpha = alpha + X(:,ix)*b(ix);                          % faster update 
                [b(ix), muBeta(ix)] = fastmvg_rue(Xscaled, [], alpha, [], Lambda(ix), sigma2, sqrt(omega2), gprior, Z);
                alpha = alpha - X(:,ix)*b(ix);
            end

        %% Gaussian regression only
        else
            for k = 1 : nBlocks
                ix = ((1:p) >= blockStart(k)) & ((1:p) <= blockEnd(k)); % current block to sample
                
                % Slower updates
                if (isempty(BlockXty_update))
                    alpha = alpha + X(:,ix)*b(ix);                          
                    [b(ix), muBeta(ix)] = fastmvg_rue([], BlockXtX{k}, [], X(:,ix)'*alpha, Lambda(ix), sigma2, sqrt(omega2), gprior, BlockXtX{k});
                    alpha = alpha - X(:,ix)*b(ix);
                    
                % Faster updates
                else
                    if (isempty(Xt1))
                        [b(ix), muBeta(ix)] = fastmvg_rue([], BlockXtX{k}, [], BlockXty{k} - BlockXty_update{k}*b(~ix), Lambda(ix), sigma2, [], gprior, BlockXtX{k});
                    else
                        [b(ix), muBeta(ix)] = fastmvg_rue([], BlockXtX{k}, [], BlockXty{k} - BlockXty_update{k}*b(~ix) - Xt1(ix)*b0, Lambda(ix), sigma2, [], gprior, BlockXtX{k});
                    end
                end
            end
        end

    %% Use Bhat. algorithm
    else
        alpha = (z - b0 - X*b);    
        
        omega = sqrt(omega2);
        for k = 1 : nBlocks
            % Current block to sample
            ix = ((1:p) >= blockStart(k)) & ((1:p) <= blockEnd(k)); 

            % Update residuals and sample block
            Xscaled = bsxfun(@rdivide, X(:,ix), omega);
            alpha = alpha + X(:,ix)*b(ix);                          
            [b(ix), muBeta(ix)] = fastmvg_bhat(Xscaled, alpha, Lambda(ix), sigma, omega);
            alpha = alpha - X(:,ix)*b(ix);
        end
    end
end

end
