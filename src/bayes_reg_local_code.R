#' Fit a linear or logistic regression model using Bayesian continuous shrinkage prior distributions. Handles ridge, lasso, horseshoe and horseshoe+ regression with logistic, # nolint: line_length_linter.
#' Gaussian, Laplace, Student-t, Poisson or geometric distributed targets. See \code{\link{bayesreg-package}} for more details on the features available in this package.
#'
#' @title Fitting Bayesian Regression Models with Continuous Shrinkage Priors
#' @param formula An object of class "\code{\link{formula}}": a symbolic description of the model to be fitted using the standard R formula notation.
#' @param data A data frame containing the variables in the model.
#' @param model The distribution of the target (y) variable. Continuous or numeric variables can be distributed as per a Gaussian distribution (\code{model="gaussian"} 
#' or \code{model="normal"}), Laplace distribution (\code{model = "laplace"} or \code{model = "l1"}) or Student-t distribution (\code{"model" = "studentt"} or \code{"model" = "t"}). 
#' Integer or count data can be distributed as per a Poisson distribution (\code{model="poisson"}) or geometric distribution (\code{model="geometric"}).
#' For binary targets (factors with two levels) either \code{model="logistic"} or \code{"model"="binomial"} should be used.
#' @param prior Which continuous shrinkage prior distribution over the regression coefficients to use. Options include ridge regression 
#' (\code{prior="rr"} or \code{prior="ridge"}), lasso regression (\code{prior="lasso"}), horseshoe regression (\code{prior="hs"} or \code{prior="horseshoe"}) and 
#' horseshoe+ regression (\code{prior="hs+"} or \code{prior="horseshoe+"})
#' @param n.samples Number of posterior samples to generate.
#' @param burnin Number of burn-in samples.
#' @param thin Desired level of thinning.
#' @param t.dof Degrees of freedom for the Student-t distribution.
#' @section Details:
#' Draws a series of samples from the posterior distribution of a linear (Gaussian, Laplace or Student-t) or generalized linear (logistic binary, Poisson, geometric) regression model with specified continuous 
#' shrinkage prior distribution (ridge regression, lasso, horseshoe and horseshoe+) using Gibbs sampling. The intercept parameter is always included, and is never penalised.
#' 
#' While only \code{n.samples} are returned, the total number of samples generated is equal to \code{burnin}+\code{n.samples}*\code{thin}. To generate the samples 
#' of the regression coefficients, the code will use either Rue's algorithm (when the number of samples is twice the number of covariates) or the algorithm of 
#' Bhattacharya et al. as appropriate. Factor variables are automatically grouped together and 
#' additional shrinkage is applied to the set of indicator variables to which they expand.
#' 
#' @return An object with S3 class \code{"bayesreg"} containing the results of the sampling process, plus some additional information.
#' \item{beta}{Posterior samples the regression model coefficients.}
#' \item{beta0}{Posterior samples of the intercept parameter.}
#' \item{sigma2}{Posterior samples of the square of the scale parameter; for Gaussian distributed targets this is equal to the variance. For binary targets this is empty.}
#' \item{mu.beta}{The mean of the posterior samples for the regression coefficients.}
#' \item{mu.beta0}{The mean of the posterior samples for the intercept parameter.}
#' \item{mu.sigma2}{The mean of the posterior samples for squared scale parameter.}
#' \item{tau2}{Posterior samples of the global shrinkage parameter.}
#' \item{t.stat}{Posterior t-statistics for each regression coefficient.}
#' \item{var.ranks}{Ranking of the covariates by their importance, with "1" denoting the most important covariate.}
#' \item{log.l}{The log-likelihood at the posterior means of the model parameters}
#' \item{waic}{The Widely Applicable Information Criterion (WAIC) score for the model}
#' \item{waic.dof}{The effective degrees-of-freedom of the model, as estimated by the WAIC.}
#' The returned object also stores the parameters/options used to run \code{bayesreg}:
#' \item{formula}{The object of type "\code{\link{formula}}" describing the fitted model.}
#' \item{model}{The distribution of the target (y) variable.}
#' \item{prior}{The shrinkage prior used to fit the model.}
#' \item{n.samples}{The number of samples generated from the posterior distribution.}
#' \item{burnin}{The number of burnin samples that were generated.}
#' \item{thin}{The level of thinning.}
#' \item{n}{The sample size of the data used to fit the model.}
#' \item{p}{The number of covariates in the fitted model.}
#' 
#' @references 
#' 
#' Makalic, E. & Schmidt, D. F.
#' High-Dimensional Bayesian Regularised Regression with the BayesReg Package
#' arXiv:1611.06649 [stat.CO], 2016 \url{https://arxiv.org/pdf/1611.06649.pdf}
#' 
#' Park, T. & Casella, G. 
#' The Bayesian Lasso 
#' Journal of the American Statistical Association, Vol. 103, pp. 681-686, 2008
#' 
#' Carvalho, C. M.; Polson, N. G. & Scott, J. G. 
#' The horseshoe estimator for sparse signals 
#' Biometrika, Vol. 97, 465-480, 2010
#' 
#' Makalic, E. & Schmidt, D. F. 
#' A Simple Sampler for the Horseshoe Estimator 
#' IEEE Signal Processing Letters, Vol. 23, pp. 179-182, 2016 \url{https://arxiv.org/pdf/1508.03884v4.pdf}
#' 
#' Bhadra, A.; Datta, J.; Polson, N. G. & Willard, B. 
#' The Horseshoe+ Estimator of Ultra-Sparse Signals 
#' Bayesian Analysis, 2016
#' 
#' Polson, N. G.; Scott, J. G. & Windle, J. 
#' Bayesian inference for logistic models using Polya-Gamma latent variables 
#' Journal of the American Statistical Association, Vol. 108, 1339-1349, 2013
#' 
#' Rue, H. 
#' Fast sampling of Gaussian Markov random fields 
#' Journal of the Royal Statistical Society (Series B), Vol. 63, 325-338, 2001
#' 
#' Bhattacharya, A.; Chakraborty, A. & Mallick, B. K. 
#' Fast sampling with Gaussian scale-mixture priors in high-dimensional regression 
#' arXiv:1506.04778, 2016
#' 
#' Schmidt, D.F. & Makalic, E.
#' Bayesian Generalized Horseshoe Estimation of Generalized Linear Models
#' ECML PKDD 2019: Machine Learning and Knowledge Discovery in Databases. pp 598-613, 2019
#' 
#' Stan Development Team, Stan Reference Manual (Version 2.26), Section 15.4, "Effective Sample Size",
#' \url{https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html}
#'    
#' @note
#' To cite this toolbox please reference: 
#'   
#' Makalic, E. & Schmidt, D. F.
#' High-Dimensional Bayesian Regularised Regression with the BayesReg Package
#' arXiv:1611.06649 [stat.CO], 2016 \url{https://arxiv.org/pdf/1611.06649.pdf}
#' 
#' A MATLAB implementation of the bayesreg function is also available from:
#' 
#' \url{https://au.mathworks.com/matlabcentral/fileexchange/60823-flexible-bayesian-penalized-regression-modelling}
#' 
#' Copyright (C) Daniel F. Schmidt and Enes Makalic, 2016-2021
#' 
#' @seealso The prediction function \code{\link{predict.bayesreg}} and summary function \code{\link{summary.bayesreg}}
#' @examples 
#' # -----------------------------------------------------------------
#' # Example 1: Gaussian regression
#' X = matrix(rnorm(100*20),100,20)
#' b = matrix(0,20,1)
#' b[1:5] = c(5,4,3,2,1)
#' y = X %*% b + rnorm(100, 0, 1)
#' 
#' df <- data.frame(X,y)
#' rv.lm <- lm(y~.,df)                        # Regular least-squares
#' summary(rv.lm)
#' 
#' rv.hs <- bayesreg(y~.,df,prior="hs")       # Horseshoe regression
#' rv.hs.s <- summary(rv.hs)
#' 
#' # Expected squared prediction error for least-squares
#' coef_ls = coef(rv.lm)
#' as.numeric(sum( (as.matrix(coef_ls[-1]) - b)^2 ) + coef_ls[1]^2)
#' 
#' # Expected squared prediction error for horseshoe
#' as.numeric(sum( (rv.hs$mu.beta - b)^2 ) + rv.hs$mu.beta0^2)
#' 
#' 
#' # -----------------------------------------------------------------
#' # Example 2: Gaussian v Student-t robust regression
#' X = 1:10;
#' y = c(-0.6867, 1.7258, 1.9117, 6.1832, 5.3636, 7.1139, 9.5668, 10.0593, 11.4044, 6.1677);
#' df = data.frame(X,y)
#' 
#' # Gaussian ridge
#' rv.G <- bayesreg(y~., df, model = "gaussian", prior = "ridge", n.samples = 1e3)
#' 
#' # Student-t ridge
#' rv.t <- bayesreg(y~., df, model = "t", prior = "ridge", t.dof = 5, n.samples = 1e3)
#' 
#' # Plot the different estimates with credible intervals
#' plot(df$X, df$y, xlab="x", ylab="y")
#' 
#' yhat_G <- predict(rv.G, df, bayes.avg=TRUE)
#' lines(df$X, yhat_G[,1], col="blue", lwd=2.5)
#' lines(df$X, yhat_G[,3], col="blue", lwd=1, lty="dashed")
#' lines(df$X, yhat_G[,4], col="blue", lwd=1, lty="dashed")
#' 
#' yhat_t <- predict(rv.t, df, bayes.avg=TRUE)
#' lines(df$X, yhat_t[,1], col="darkred", lwd=2.5)
#' lines(df$X, yhat_t[,3], col="darkred", lwd=1, lty="dashed")
#' lines(df$X, yhat_t[,4], col="darkred", lwd=1, lty="dashed")
#' 
#' legend(1,11,c("Gaussian","Student-t (dof=5)"),lty=c(1,1),col=c("blue","darkred"),
#'        lwd=c(2.5,2.5), cex=0.7)
#' 
#' \dontrun{
#' # -----------------------------------------------------------------
#' # Example 3: Poisson/geometric regression example
#' 
#' X  = matrix(rnorm(100*20),100,5)
#' b  = c(0.5,-1,0,0,1)
#' nu = X%*%b + 1
#' y  = rpois(lambda=exp(nu),n=length(nu))
#'
#' df <- data.frame(X,y)
#'
#' # Fit a Poisson regression
#' rv.pois=bayesreg(y~.,data=df,model="poisson",prior="hs", burnin=1e4, n.samples=1e4)
#' summary(rv.pois)
#' 
#' # Fit a geometric regression
#' rv.geo=bayesreg(y~.,data=df,model="geometric",prior="hs", burnin=1e4, n.samples=1e4)
#' summary(rv.geo)
#' 
#' # Compare the two models in terms of their WAIC scores
#' cat(sprintf("Poisson regression WAIC=%g vs geometric regression WAIC=%g", 
#'             rv.pois$waic, rv.geo$waic))
#' # Poisson is clearly preferred to geometric, which is good as data is generated from a Poisson!
#'  
#'  
#' # -----------------------------------------------------------------
#' # Example 4: Logistic regression on spambase
#' data(spambase)
#'   
#' # bayesreg expects binary targets to be factors
#' spambase$is.spam <- factor(spambase$is.spam)
#' 
#' # First take a subset of the data (1/10th) for training, reserve the rest for testing
#' spambase.tr  = spambase[seq(1,nrow(spambase),10),]
#' spambase.tst = spambase[-seq(1,nrow(spambase),10),]
#'   
#' # Fit a model using logistic horseshoe for 2,000 samples
#' rv <- bayesreg(is.spam ~ ., spambase.tr, model = "logistic", prior = "horseshoe", n.samples = 2e3)
#'   
#' # Summarise, sorting variables by their ranking importance
#' rv.s <- summary(rv,sort.rank=TRUE)
#'   
#' # Make predictions about testing data -- get class predictions and class probabilities
#' y_pred <- predict(rv, spambase.tst, type='class')
#'   
#' # Check how well did our predictions did by generating confusion matrix
#' table(y_pred, spambase.tst$is.spam)
#'   
#' # Calculate logarithmic loss on test data
#' y_prob <- predict(rv, spambase.tst, type='prob')
#' cat('Neg Log-Like for no Bayes average, posterior mean estimates: ', sum(-log(y_prob[,1])), '\n')
#' y_prob <- predict(rv, spambase.tst, type='prob', sum.stat="median")
#' cat('Neg Log-Like for no Bayes average, posterior median estimates: ', sum(-log(y_prob[,1])), '\n')
#' y_prob <- predict(rv, spambase.tst, type='prob', bayes.avg=TRUE)
#' cat('Neg Log-Like for Bayes average: ', sum(-log(y_prob[,1])), '\n')
#' }
#' 
#' @export
#' 
#' 
bayesreg_local <- function(formula, data, model='normal', prior='ridge', n.samples = 1e3, burnin = 1e3, thin = 5, t.dof = 5, std = FALSE)
{
  VERSION = '1.2'
  groups  = NA
  display = F
  rankvars = T
  
  printf <- function(...) cat(sprintf(...))
  
  # Return object
  rv = list()

  # -------------------------------------------------------------------    
  # model/target distribution types
  SMN   = T
  MH    = F
  model = tolower(model)
  if (model == 'normal' || model == 'gaussian')
  {
    model = 'gaussian'
  }
  else if (model == 'laplace' || model == 'l1')
  {
    model = 'laplace'
  }
  else if (model == 'studentt' || model == 't')
  {
    model = 't'
  }
  else if (model == 'logistic' || model == "binomial")
  {
    model = 'logistic'
  }
  else if (model == 'poisson' || model == 'geometric')
  {
    SMN = F
    MH  = T
  }
  else
  {
    stop('Unknown target distribution \'',model,'\'')
  }
  
  # -------------------------------------------------------------------    
  # Process and set up the data from the model formula
  rv$terms <- stats::terms(x = formula, data = data)
  
  mf = stats::model.frame(formula = formula, data = data)
  rv$target.var = names(mf)[1]
  if (model == 'logistic')
  {
    # Check to ensure target is a factor
    if (!is.factor(mf[,1]) || (!is.factor(mf[,1]) && length(levels(mf[,1])) != 2))
    {
      stop('Target variable must be a factor with two levels for logistic regression')
    }
    
    rv$ylevels <- levels(mf[,1])
    mf[,1] = as.numeric(mf[,1])
    mf[,1] = (as.numeric(mf[,1]) - 1)
  }
  
  # If count regression, check targets are counts
  else if (model == 'poisson' || model == 'geometric')
  {
    if (any((floor(mf[,1]) != mf[,1])) || any(mf[,1]<0))
    {
      stop('Target variable must be non-negative integers for count regression')
    }
  }
  
  # Otherwise check to ensure target is numeric
  else if (!is.numeric(mf[,1]))
  {
    stop('Target variable must not be a factor for linear regression; use model = "logistic" instead')
  }
  
  # If not logistic, check if targets have only 2 unique values -- if so, give a warning
  if (model != 'logistic' && length(unique(mf[,1])) == 2)
  {
    warning('Target variable takes on only two distinct values -- should this be a binary regression?')
  }
  
  y = mf[,1]
  X = stats::model.matrix(formula, data=data)
  
  # Assign factors to groups if categorical variables have > 2 levels
  groups = matrix(NA,ncol(X),1)
  gnum = 1
  cn = colnames(X)
  assign = attr(X, "assign")
  for (j in 2:ncol(mf))
  {
    # If it is a factor, set up the groups as appropriate
    if (is.factor(mf[,j]) && length(levels(mf[,j]))>2)
    {
      groups[(assign==(j-1))] = gnum
      gnum = gnum+1
    }
  }
  groups = groups[2:length(groups)]  

  # Convert to a numeric matrix and drop the target variable
  X = as.matrix(X)
  X = X[,-1,drop=FALSE]
  
  # 
  n = nrow(X)
  p = ncol(X)

  # -------------------------------------------------------------------
  # Prior types
  if (prior == 'ridge' || prior == 'rr')
  {
    prior = 'rr'
  }
  else if (prior == 'horseshoe' || prior == 'hs')
  {
    prior = 'hs'
  }
  else if (prior == 'horseshoe+' || prior == 'hs+')
  {
    prior = 'hs+'
  }
  else if (prior != 'lasso')
  {
    stop('Unknown prior \'', prior, '\'')
  }
  
  # -------------------------------------------------------------------
  # Standardise data?
  std.X = bayesreg.standardise(X)
  if (std == TRUE) {
    X <- std.X$X
  }

  # Initial values
  ydiff    = 0
  b0       = 0
  b        = matrix(0,p,1)
  omega2   = matrix(1,n,1)
  sigma2   = 1

  tau2     = 1
  xi       = 1
  
  lambda2  = matrix(1,p,1)
  nu       = matrix(1,p,1)

  eta2     = matrix(1,p,1)
  phi      = matrix(1,p,1)

  kappa    = y - 1/2
  z        = y

  # Quantities for computing WAIC 
  waicProb   = matrix(0,n,1) 
  waicLProb  = matrix(0,n,1)
  waicLProb2 = matrix(0,n,1)
  
  # Use Rue's MVN sampling algorithm
  mvnrue = T
  PrecomputedXtX = F
  if (p/n >= 2)
  {
    # Switch to Bhatta's MVN sampling algorithm
    mvnrue = F
  }
  
  # Precompute XtX?
  XtX  = NA
  Xty  = NA
  
  if (model == "gaussian" && mvnrue)
  {
    XtX = crossprod(X)
    Xty = crossprod(X,y)
    PrecomputedXtX = T
  }
  
  # Metropolis-Hastings gradient based sampling setup
  if (MH)
  {
    # If insufficient burn-in available
    if (burnin < 1e3)
    {
      stop('To use Metropolis-Hastings sampling you must use at least 1,000 burnin samples')
    }
    
    # Initialise betas and b0 with rough ridge solutions
    # Poisson
    if (model == 'poisson')
    {
      extra.model.params = NULL
      
      Xty = crossprod(X,y)
      rv.gd = bayesreg.fit.GLM.gd(X, y, model, 1, 10, 5e2)
      rv.mh = bayesreg.mgrad.L(y, X, as.vector(c(rv.gd$b,rv.gd$b0)), NULL, model, extra.model.params, Xty)
    }
    if (model == 'geometric')
    {
      extra.model.params = 1
      
      Xty = crossprod(X,y)
      rv.gd = bayesreg.fit.GLM.gd(X, y, model, 1, 10, 1e3)
      rv.mh = bayesreg.mgrad.L(y, X, as.vector(c(rv.gd$b,rv.gd$b0)), NULL, model, extra.model.params, Xty)
    }
    
    # Initialise variables
    b      = rv.gd$b
    b0     = rv.gd$b0
    L.b    = rv.mh$L
    grad.b = rv.mh$grad[1:p]
    eta    = rv.mh$eta
    H.b0   = rv.mh$H.b0
    
    extra.stats = NULL
    
    mh.tune = bayesreg.mh.initialise(75, 1e2, 1e-7, burnin, T)
    mh.tune$delta = 0.0002
  }

  # Set up group shrinkage parameters if required
  ng = 0
  if (length(groups) > 1)
  {
    ngroups = length(unique(groups))
    groups[is.na(groups)] = ngroups
    delta2  = matrix(1,ngroups,1)
    chi     = matrix(1,ngroups,1)
    rho2    = matrix(1,ngroups,1)
    zeta    = matrix(1,ngroups,1)
    ngroups = ngroups-1
  }
  else
  {
    ngroups = 1
    groups  = matrix(1,1,p)
    delta2  = matrix(1,1,1)
    chi     = matrix(1,1,1)
    rho2    = matrix(1,1,1)
    zeta    = matrix(1,1,1)
  }

  # Setup return object
  rv$formula   = formula
  rv$model     = model
  rv$prior     = prior
  rv$n.samples = n.samples
  rv$burnin    = burnin
  rv$thin      = thin
  rv$groups    = groups
  rv$t.dof     = t.dof
  
  rv$n         = n
  rv$p         = p
  rv$tss       = sum((y - mean(y))^2)

  rv$beta0     = matrix(0,1,n.samples)
  rv$beta      = matrix(0,p,n.samples)
  rv$mu.beta   = matrix(0,p,1)
  rv$mu.beta0  = matrix(0,1,1)
  rv$sigma2    = matrix(0,1,n.samples)
  rv$mu.sigma2 = matrix(0,1,1)
  rv$tau2      = matrix(0,1,n.samples)
  #rv$xi       = matrix(0,1,n.samples)
  #rv$lambda2  = matrix(0,p,n.samples)
  #rv$nu       = matrix(0,p,n.samples)
  #rv$delta2   = matrix(0,ngroups,n.samples)
  #rv$chi      = matrix(0,ngroups,n.samples)
  
  rv$t.stat    = matrix(0,p,1)

  # Print the banner  
  if (display)
  {
    printf('==========================================================================================\n');
    printf('|                   Bayesian Penalised Regression Estimation ver. %s                    |\n', VERSION);
    printf('|                     (c) Enes Makalic, Daniel F Schmidt. 2016-2021                      |\n');
    printf('==========================================================================================\n');
  }

  # Main sampling loop
  k    = 0
  iter = 0
  while (k < n.samples)
  {
    # If using scale-mixture of normals sampler
    if (SMN)
    {
      # ===================================================================
      # Sample regression coefficients (beta)
      if (model == 'logistic')
      {
        z = kappa * omega2
      }
  
      bs  = bayesreg.sample_beta(X, z, mvnrue, b0, sigma2, tau2, lambda2 * delta2[groups], omega2, XtX, Xty, model, PrecomputedXtX)
      muB = bs$m
      b   = bs$x
   
      # ===================================================================
      # Sample intercept (b0)
      W         = sum(1 / omega2)
      e         = y - X %*% b
      
      # Non-logistic models
      if (model != 'logistic')
      {
        muB0    = sum(e / omega2) / W
      }
      else
      {
        W       = sum(1 / omega2)
        muB0    = sum((z-(y-e)) / omega2) / W
      }
      v       = sigma2 / W
      
      # Sample b0 and update residuals
      b0      = stats::rnorm(1, muB0, sqrt(v))    
      e       = e-b0
  
      # ===================================================================
      # Sample the noise scale parameter sigma2
      mu.sigma2   = NA
      if (model != 'logistic')
      {
        #e         = y - X %*% b - b0
        shape    = (n + p)/2
        scale    = sum( (e^2 / omega2)/2 ) + sum( (b^2 / delta2[groups] / lambda2 / tau2)/2 )
        sigma2   = 1/stats::rgamma(1, shape=shape, scale=1/scale)
        
        mu.sigma2 = scale / (shape-1)
      }
    }
    # Else if using Metropolis-Hastings
    else if (MH)
    {
      mu.sigma2 = 1
      
      # Sample beta's
      mh.tune$D = mh.tune$D + 1
      
      rv.mh = bayesreg.mgrad.sample.beta(b, b0, L.b, grad.b, sigma2*tau2*lambda2*delta2[groups], mh.tune$delta, eta, H.b0, y, X, sigma2, Xty, model, extra.model.params, extra.stats)
      
      # If it was accepted?
      if (rv.mh$accepted)
      {
        mh.tune$M = mh.tune$M + 1
        
        b      = rv.mh$b
        b0     = rv.mh$b0
        L.b    = rv.mh$L.b
        grad.b = rv.mh$grad.b
        eta    = rv.mh$eta
        H.b0   = rv.mh$H.b0
      }
      
      # Update as required
      muB = rv.mh$b
      muB0 = rv.mh$b0
      
      # Sample b0 using simple MH sampler (only every 25 ticks)
      if (iter %% 25 == 0)
      {
        b0.new = rnorm(n=1, mean=b0, sd=sqrt(2.5/rv.mh$H.b0))
        if (model == "poisson" || model == "geometric")
        {
          rv.mh = bayesreg.mgrad.L(y, X, c(b,b0), eta-b0+b0.new, model, extra.model.params, Xty)
          #[L_bnew, grad_bnew, H_b0new, ~, extra_stats_new] = br_mGradL(y, X, [b;b0], eta-b0+b0_new, model, extra_model_params, Xty);
        }
        
        # Accept?
        if (runif(1) < exp(-rv.mh$L/sigma2 + L.b/sigma2))
        {
          eta = eta - b0 + b0.new
          b0 = b0.new;
          L.b = rv.mh$L
          H_b0 = rv.mh$H.b0;
          grad.b = rv.mh$grad[1:p]
          extra.stats = rv.mh$extra.stats
        }
      }
    }
    
    # ===================================================================
    # Sample 'omega's
    
    # -------------------------------------------------------------------
    # Laplace
    if (model == 'laplace')
    {
      #e = y - X %*% b - b0
      mu = sqrt(2 * sigma2 / e^2)
      omega2 = 1 / bayesreg.rinvg(mu,1/2)
    }

    # -------------------------------------------------------------------
    # Student-t
    if (model == 't')
    {
      #e = y - X %*% b - b0
      shape = (t.dof+1)/2
      scale = (e^2/sigma2+t.dof)/2
      omega2 = as.matrix(1 / stats::rgamma(n, shape=shape, scale=1/scale), n, 1)
    }

    # -------------------------------------------------------------------
    # Logistic
    if (model == 'logistic')
    {
      #omega2 = 1 / rpg.devroye(num = n, n = 1, z = b0 + X %*% b)
      omega2 = as.matrix(1 / pgdraw::pgdraw(1, b0+X%*%b), n, 1)
    }

    # ===================================================================
    # Sample the global shrinkage parameter tau2 (and L.V. xi)
    # hs/hs+/ridge use tau2 ~ C+(0,1)

    # this nmeed to be fixed if data is not Z-scored aka standardized! 
    if (prior == 'hs' || prior == 'hs+' || prior == 'rr')
    {
      shape = (p+1)/2
      scale = 1/xi + sum(b^2 / lambda2 / delta2[groups]) / 2 / sigma2
      tau2  = 1 / stats::rgamma(1, shape=shape, scale=1/scale)
  
      # Sample xi
      scale = 1 + 1/tau2
      xi    = scale / stats::rexp(1)
    }
    
    # Lasso uses tau2 ~ IG(1,1)
    else if (prior == 'lasso')
    {
      shape = p/2+1
      scale = 1 + sum(b^2 / lambda2 / delta2[groups]) / 2 / sigma2
      tau2  = 1 / stats::rgamma(1, shape=shape, scale=1/scale)
    }
    
    # ===================================================================
    # Sample the lambda2's/nu's (if horseshoe, horseshoe+, horseshoe-grouped)
    if (prior == 'hs' || prior == 'hs+' || prior == 'hsge')
    {
      # Sample nu -- horseshoe
      if (prior == 'hs' || prior == 'hsge')
      {
        # Sample lambda2
        scale   = 1/nu + b^2 / 2 / tau2 / sigma2 / delta2[groups]
        lambda2 = scale / stats::rexp(p)

        scale = 1 + 1/lambda2
        nu    = scale / stats::rexp(p)
      }
      
      # Parameter expanded HS+ sampler
      else if (prior == 'hs+')
      {
        # Sample lambda2
        scale   = 1/nu + b^2 / 2 / tau2 / sigma2 / (delta2[groups]*eta2)
        lambda2 = scale / stats::rexp(p)

        # Sample nu
        scale = 1 + 1/lambda2
        nu    = scale / stats::rexp(p)
        
        # Sample eta2
        scale =  1/phi + b^2 / 2 / tau2 / sigma2 / (delta2[groups]*lambda2)
        eta2  = scale / stats::rexp(p)
        
        # Sample phi
        scale = 1 + 1/eta2
        phi   = scale / stats::rexp(p)
        
        lambda2 = lambda2*eta2
      }
    }
    
    
    # ===================================================================
    # Sample the lambda2's (if lasso)
    if (prior == 'lasso')
    {
      mu      = sqrt(2 * sigma2 * tau2 / (b^2 / delta2[groups]))
      lambda2 = 1 / bayesreg.rinvg(mu, 1/2)
    }
    
    
    # ===================================================================
    # Sample the delta2's (if grouped horseshoe, horseshoe+ or lasso)
    if (ngroups > 1)
    {
      # Sample delta2's
      for (i in 1:ngroups)
      {
        # Only sample delta2 for this group, if the group size is > 1
        ng = sum(groups == i)
        if (ng > 1)
        {
          # Grouped horseshoe/horseshoe+
          if (prior == 'hs')
          {  
            # Sample delta2
            shape = (ng+1)/2
            scale = 1 / chi[i] + sum((b[groups==i]^2) / lambda2[groups==i]) / 2 / sigma2 / tau2
            delta2[i] = 1 / stats::rgamma(1, shape=shape, scale=1/scale)
  
            # Sample chi
            scale  = 1 + 1/delta2[i]
            chi[i] = 1 / stats::rgamma(1, shape=1, scale=1/scale)
          }
          
          # Grouped horseshoe+
          else if (prior == 'hs+')
          {
            # Sample delta2
            shape = (ng+1)/2
            scale = 1 / chi[i] + sum((b[groups==i]^2) / 2 / sigma2 / tau2 / lambda2[groups==i] / rho2[i])
            delta2[i] = 1 / stats::rgamma(1, shape=shape, scale=1/scale)
            
            # Sample chi
            scale  = 1 + 1/delta2[i]
            chi[i] = scale / stats::rexp(1)
            
            # Sample rho2
            shape = (ng+1)/2
            scale =  1 / zeta[i] + sum((b[groups==i]^2) / 2 / sigma2 / tau2 / lambda2[groups==i] / delta2[i])
            rho2[i] = 1 / stats::rgamma(1, shape=shape, scale=1/scale)
            
            # Sample zeta
            scale = 1 + 1/rho2[i]
            zeta[i] = scale / stats::rexp(1)
            
            delta2[i] = delta2[i]*rho2[i]
          }
          
          # Grouped lasso
          else if (prior == 'lasso')
          {
            mu = sqrt(2 * sigma2 * tau2 / sum((b[groups==i]^2)/lambda2[groups==i]))
            delta2[i] = 1 / bayesreg.rinvg(mu, 0.5)
          }
        }
      }
    }
    
    # ===================================================================
    # Store the samples
    iter = iter+1
    if (iter > burnin)
    {
      # thinning
      if (!(iter %% thin))
      {
        # Store posterior samples
        k = k+1
        rv$beta0[k]     = b0
        rv$beta[,k]     = b
        rv$mu.beta      = rv$mu.beta + muB
        rv$mu.beta0     = rv$mu.beta0 + muB0
        rv$sigma2[k]    = sigma2
        rv$mu.sigma2    = rv$mu.sigma2 + mu.sigma2
        rv$tau2[k]      = tau2
        #rv$xi[k]       = xi
        #rv$lambda2[,k] = lambda2
        #rv$nu[,k]      = nu
        #rv$delta2[,k]  = delta2
        #rv$chi[,k]     = chi
        
        # Neg-log-likelihood scores for WAIC
        # Linear regression (Gaussian, t, Laplace)
        if (model != 'logistic' && model != 'poisson' && model != 'geometric')
        {
          rv.ll = bayesreg.linregnlike_e(model, e, sigma2, t.dof = t.dof)
        }
        # Logistic regression
        else if (model == 'logistic')
        {
          rv.ll = bayesreg.logregnlike_eta(y, y-e)
        }
        # Count regression
        else
        {
          rv.ll = bayesreg.countregnlike_eta(model, y, eta)
        }

        waicProb   = waicProb + rv.ll$prob
        waicLProb  = waicLProb + rv.ll$negll
        waicLProb2 = waicLProb2 + rv.ll$negll^2
      }
    }
    
    # Else we are in the burnin phase; if we are using MH sampler we need to perform step-size tuning
    else if (iter <= burnin && MH)
    {
      # Tuning step
      mh.tune = bayesreg.mgrad.tune(mh.tune)
    }
  }
  
  #
  rv$mu.beta   = rv$mu.beta / n.samples
  rv$mu.beta0  = rv$mu.beta0 / n.samples
  rv$mu.sigma2 = rv$mu.sigma2 / n.samples
  
  # ===================================================================
  # Rank features, if requested
  if (rankvars)
  {
    # Run the BFR
    ranks = bayesreg.bfr(rv)

    # Determine the 75th percentile
    rv$var.ranks = rep(NA,p+1)
    q = apply(ranks,1,function(x) stats::quantile(x,0.75))
    O = sort(q,index.return = T)
    O = O$ix
    
    j = 1
    k = 1
    for (i in 1:p)
    {
      if (i >= 2)
      {
        if (q[O[i]] != q[O[i-1]])
        {
          j = j+k
          k = 1
        }
        else
        {
          k = k+1
        }
      }
      rv$var.ranks[O[i]] = j
    }
  }
  else
  {
    rv$var.ranks = rep(NA, p+1)
  }
  
  # ===================================================================
  # Compute the t-statistics
  for (i in 1:p)
  {
    rv$t.stat[i] = rv$mu.beta[i] / stats::sd(rv$beta[i,])
  }
  
  # ===================================================================
  # Compute other model statistics
  rv$yhat    = as.matrix(X) %*% matrix(rv$mu.beta, p, 1) + as.numeric(rv$mu.beta0)
  rv$r2      = 1 - sum((y - rv$yhat)^2)/rv$tss
  rv$rootmse = mean((y - rv$yhat)^2)

  # ===================================================================
  # Compute WAIC score and log-likelihoods
  rv$waic.dof = sum(waicLProb2/n.samples) - sum((waicLProb/n.samples)^2)
  rv$waic     = -sum(log(waicProb/n.samples)) + rv$waic.dof

  if (model == 'logistic')
  {
    rv$log.l  = -bayesreg.logregnlike(X, y, rv$mu.beta, rv$mu.beta0)
    rv$log.l0 = -bayesreg.logregnlike(X, y, matrix(0,p,1), log(sum(y)/(n-sum(y))))
  }
  # Count regression
  else if (model == 'poisson' || model == 'geometric')
  {
    rv$log.l  = -bayesreg.countregnlike(model, as.matrix(X), as.matrix(y), rv$mu.beta, rv$mu.beta0)
    rv$log.l0 = -bayesreg.countregnlike(model, X, y, matrix(0,p,1), log(mean(y)))
    
    # Compute overdispersion as well
    mu.eta = as.matrix(X) %*% as.matrix(rv$mu.beta) + as.numeric(rv$mu.beta0)
    if (model == 'poisson')
    {
      rv$over.dispersion = mean( (y-exp(mu.eta))^2 / exp(mu.eta) )
    }
    else
    {
      geo.p = 1/(exp(mu.eta)+1)
      rv$over.dispersion = mean( (y-exp(mu.eta))^2 / ((1-geo.p)/geo.p^2) )
    }
  }
  # Otherwise continuous targets
  else
  {
    rv$log.l  = -bayesreg.linregnlike(model, as.matrix(X), as.matrix(y), rv$mu.beta, rv$mu.beta0, rv$mu.sigma2, rv$t.dof)
  }

  # ===================================================================
  # Rescale the coefficients
  if (p == 1)
  {
    rv$beta  <- t(as.matrix(apply(t(rv$beta), 1, function(x)(x / std.X$std.X))))
  }
  else
  {
    rv$beta  <- as.matrix(apply(t(rv$beta), 1, function(x)(x / std.X$std.X)))
  }
  rv$beta0 <- rv$beta0 - std.X$mean.X %*% rv$beta
  
  rv$mu.beta  <- rv$mu.beta / t(std.X$std.X)
  rv$mu.beta0 <- rv$mu.beta0 - std.X$mean.X %*% rv$mu.beta

  rv$std.X = std.X
  
  # ===================================================================
  # Compute effective sample sizes
  rv$ess.frac = rep(NA, p)
  for (i in 1:p)
  {
    e = bayesreg.ess(rv$beta[i,])
    rv$ess.frac[i] = e$ess.frac
  }
  rv$ess.frac[ rv$ess.frac > 1 ] = 1
  rv$ess = ceiling(rv$ess.frac * n.samples)

  class(rv) = "bayesreg"
  return(rv)
}


# ============================================================================================================================
# Sample the intercept
bayesreg.sample_beta0 <- function(X, z, b, sigma2, omega2)
{
  rv      = list()
  W       = sum(1 / omega2)
  rv$muB0 = sum((z - X %*% b) / omega2) / W
  v       = sigma2 / W
  
  rv$b0  = stats::rnorm(1, rv$muB0, sqrt(v))

  # Done
  rv
}


# ============================================================================================================================
# Compute the effective sample size of a sequence of RVs using the algorithm from the Stan user manual
bayesreg.ess <- function(x)
{
  n = length(x)
  s = min(c(n - 1, 2e4))
  g = stats::acf(x, s, plot=F)

  i = seq(1,s-1,by=2)
  G = as.vector(g$acf[i]) + as.vector(g$acf[i+1])
  for (i in 2:length(G))
  {
    if (G[i] > G[i-1])
    {
      G[i] = G[i-1]
    }
  }
  Gz = G<0
  if (sum(Gz) == 0)
  {
    k = length(G)
  }
  else
  {
    for (i in 1:length(G))
    {
      if (Gz[i])
      {
        break
      }
    }
    k = i
  }

  rv = list()
  rv$ess = n/(-1 + 2*sum(G[1:k]))
  rv$ess.frac = rv$ess/n

  # G = as.vector(g$acf[2:(s-1)]) + as.vector(g$acf[3:s])
  # G = G < 0
  # for (i in 1:length(G))
  # {
  #   if (G[i])
  #   {
  #     break
  #   }
  # }
  # k = i
  # if (k >= 2)
  #   V = g$acf[1] + 2 * sum(g$acf[2:i])
  # else
  #   V = g$acf[1]
  # 
  # ACT = V / g$acf[1]
  # rv = list()
  # rv$ESS = n/ ACT
  # rv$ess.frac = rv$ESS / n
  
  return(rv)
}


# ============================================================================================================================
# Bayesian Feature Ranking
bayesreg.bfr <- function(rv)
{
  ranks = matrix(0, rv$p, rv$n.samples)
  
  for (i in 1:rv$n.samples)
  {
    r = sort(-abs(rv$beta[,i]), index.return = T)
    ranks[r$ix,i] = 1:rv$p
  }
  
  return(ranks)
}   


#' Predict values based on Bayesian penalised regression (\code{\link{bayesreg}}) models.
#'
#' @title Prediction method for Bayesian penalised regression (\code{bayesreg}) models
#' @param object an object of class \code{"bayesreg"} created as a result of a call to \code{\link{bayesreg}}.
#' @param newdata A data frame providing the variables from which to produce predictions.
#' @param type The type of predictions to produce; if \code{type="linpred"} it will return the linear predictor for binary, 
#' count and continuous data. If \code{type="prob"} it will return predictive probability estimates for provided 'y' data (see below for more details).
#' If \code{type="response"} it will return the predicted conditional mean of the target (see below for more details).
#' If \code{type="class"} and the data is binary, it will return the best guess at the class of the target variable.
#' @param bayes.avg logical; whether to produce predictions using Bayesian averaging.
#' @param sum.stat The type of summary statistic to use; either \code{sum.stat="mean"} or \code{sum.stat="median"}.
#' @param CI The size (level, as a percentage) of the credible interval to report (default: 95, i.e. a 95\% credible interval)
#' @param ... Further arguments passed to or from other methods.
#' @section Details:
#' \code{predict.bayesreg} produces predicted values using variables from the specified data frame. The type of predictions produced 
#' depend on the value of the parameter \code{type}:
#'
#' \itemize{
#' \item If \code{type="linpred"}, the predictions that are returned will be the value of the linear predictor formed from the model 
#' coefficients and the provided data. 
#' 
#' \item If \code{type="response"}, the predictions will be the conditional mean for each data point. For Gaussian, Laplace and Student-t targets
#' the conditional mean is simply equal to the linear predictor; for binary data, the predictions will 
#' be the probability of the target being equal to the second level of the factor variable; for count data, the conditional mean
#' will be exp(linear predictor).
#' 
#' \item If \code{type="prob"}, the predictions will be probabilities. The specified data frame must include a column with the same name as the 
#' target variable on which the model was created. The predictions will then be the probability (density) values for these target values. 
#' 
#' \item If \code{type="class"} and the target variable is binary, the predictions will be the most likely class.
#' }
#' 
#' If \code{bayes.avg} is \code{FALSE} the predictions will be produced by using a summary of the posterior samples of the coefficients 
#' and scale parameters as estimates for the model. If \code{bayes.avg} is \code{TRUE}, the predictions will be produced by posterior 
#' averaging over the posterior samples of the coefficients and scale parameters, allowing the uncertainty in the estimation process to 
#' be explicitly taken into account in the prediction process. 
#' 
#' If \code{sum.stat="mean"} and \code{bayes.avg} is \code{FALSE}, the mean of the posterior samples will be used as point estimates for
#' making predictions. Likewise, if \code{sum.stat="median"} and \code{bayes.avg} is \code{FALSE}, the co-ordinate wise posterior medians 
#' will be used as estimates for making predictions. If \code{bayes.avg} is \code{TRUE} and \code{type!="prob"}, the posterior mean 
#' (median) of the predictions from each of the posterior samples will be used as predictions. The value of \code{sum.stat} has no effect 
#' if \code{type="prob"}.
#' @return 
#' \code{predict.bayesreg} produces a vector or matrix of predictions of the specified type. If \code{bayes.avg} is 
#' \code{FALSE} a matrix with a single column \code{pred} is returned, containing the predictions.
#'
#' If \code{bayes.avg} is \code{TRUE}, three additional columns are returned: \code{se(pred)}, which contains 
#' standard errors for the predictions, and two columns containing the credible intervals (at the specified level) for the predictions.
#' @seealso The model fitting function \code{\link{bayesreg}} and summary function \code{\link{summary.bayesreg}}
#' @examples
#' 
#' # -----------------------------------------------------------------
#' # Example 1: Fitting linear models to data and generating credible intervals
#' X = 1:10;
#' y = c(-0.6867, 1.7258, 1.9117, 6.1832, 5.3636, 7.1139, 9.5668, 10.0593, 11.4044, 6.1677);
#' df = data.frame(X,y)
#' 
#' # Gaussian ridge
#' rv.L <- bayesreg(y~., df, model = "laplace", prior = "ridge", n.samples = 1e3)
#' 
#' # Plot the different estimates with credible intervals
#' plot(df$X, df$y, xlab="x", ylab="y")
#' 
#' yhat <- predict(rv.L, df, bayes.avg=TRUE)
#' lines(df$X, yhat[,1], col="blue", lwd=2.5)
#' lines(df$X, yhat[,3], col="blue", lwd=1, lty="dashed")
#' lines(df$X, yhat[,4], col="blue", lwd=1, lty="dashed")
#' yhat <- predict(rv.L, df, bayes.avg=TRUE, sum.stat = "median")
#' lines(df$X, yhat[,1], col="red", lwd=2.5)
#' 
#' legend(1,11,c("Posterior Mean (Bayes Average)","Posterior Median (Bayes Average)"),
#'        lty=c(1,1),col=c("blue","red"),lwd=c(2.5,2.5), cex=0.7)
#' 
#' 
#' # -----------------------------------------------------------------
#' # Example 2: Predictive density for continuous data
#' X = 1:10;
#' y = c(-0.6867, 1.7258, 1.9117, 6.1832, 5.3636, 7.1139, 9.5668, 10.0593, 11.4044, 6.1677);
#' df = data.frame(X,y)
#' 
#' # Gaussian ridge
#' rv.G <- bayesreg(y~., df, model = "gaussian", prior = "ridge", n.samples = 1e3)
#' 
#' # Produce predictive density for X=2
#' df.tst = data.frame(y=seq(-7,12,0.01),X=2)
#' prob_noavg_mean <- predict(rv.G, df.tst, bayes.avg=FALSE, type="prob", sum.stat = "mean")
#' prob_noavg_med  <- predict(rv.G, df.tst, bayes.avg=FALSE, type="prob", sum.stat = "median")
#' prob_avg        <- predict(rv.G, df.tst, bayes.avg=TRUE, type="prob")
#' 
#' # Plot the density
#' plot(NULL, xlim=c(-7,12), ylim=c(0,0.14), xlab="y", ylab="p(y)")
#' lines(df.tst$y, prob_noavg_mean[,1],lwd=1.5)
#' lines(df.tst$y, prob_noavg_med[,1], col="red",lwd=1.5)
#' lines(df.tst$y, prob_avg[,1], col="green",lwd=1.5)
#' 
#' legend(-7,0.14,c("Mean (no averaging)","Median (no averaging)","Bayes Average"),
#'        lty=c(1,1,1),col=c("black","red","green"),lwd=c(1.5,1.5,1.5), cex=0.7)
#' 
#' title('Predictive densities for X=2')
#' 
#' 
#' \dontrun{
#' # -----------------------------------------------------------------
#' # Example 3: Poisson (count) regression
#' 
#' X  = matrix(rnorm(100*20),100,5)
#' b  = c(0.5,-1,0,0,1)
#' nu = X%*%b + 1
#' y  = rpois(lambda=exp(nu),n=length(nu))
#'
#' df <- data.frame(X,y)
#'
#' # Fit a Poisson regression
#' rv.pois = bayesreg(y~.,data=df, model="poisson", prior="hs", burnin=1e4, n.samples=1e4)
#'  
#' # Make a prediction for the first five rows
#' # By default this predicts the log-rate (i.e., the linear predictor)
#' predict(rv.pois,df[1:5,]) 
#' 
#' # This is the response (i.e., conditional mean of y)
#' exp(predict(rv.pois,df[1:5,])) 
#' 
#' # Same as above ... compare to the actual targets
#' cbind(exp(predict(rv.pois,df[1:5,])), y[1:5])
#' 
#' # Slightly different as E[exp(x)]!=exp(E[x])
#' predict(rv.pois,df[1:5,], type="response", bayes.avg=TRUE) 
#' 
#' # 99% credible interval for response
#' predict(rv.pois,df[1:5,], type="response", bayes.avg=TRUE, CI=99) 
#' 
#' 
#' # -----------------------------------------------------------------
#' # Example 4: Logistic regression on spambase
#' data(spambase)
#'  
#' # bayesreg expects binary targets to be factors
#' spambase$is.spam <- factor(spambase$is.spam)
#'   
#' # First take a subset of the data (1/10th) for training, reserve the rest for testing
#' spambase.tr  = spambase[seq(1,nrow(spambase),10),]
#' spambase.tst = spambase[-seq(1,nrow(spambase),10),]
#'   
#' # Fit a model using logistic horseshoe for 2,000 samples
#' rv <- bayesreg(is.spam ~ ., spambase.tr, model = "logistic", prior = "horseshoe", n.samples = 2e3)
#'   
#' # Summarise, sorting variables by their ranking importance
#' rv.s <- summary(rv,sort.rank=TRUE)
#' 
#' # Make predictions about testing data -- get class predictions and class probabilities
#' y_pred <- predict(rv, spambase.tst, type='class')
#' y_prob <- predict(rv, spambase.tst, type='prob')
#' 
#' # Check how well our predictions did by generating confusion matrix
#' table(y_pred, spambase.tst$is.spam)
#' 
#' # Calculate logarithmic loss on test data
#' y_prob <- predict(rv, spambase.tst, type='prob')
#' cat('Neg Log-Like for no Bayes average, posterior mean estimates: ', sum(-log(y_prob[,1])), '\n')
#' y_prob <- predict(rv, spambase.tst, type='prob', sum.stat="median")
#' cat('Neg Log-Like for no Bayes average, posterior median estimates: ', sum(-log(y_prob[,1])), '\n')
#' y_prob <- predict(rv, spambase.tst, type='prob', bayes.avg=TRUE)
#' cat('Neg Log-Like for Bayes average: ', sum(-log(y_prob[,1])), '\n')
#' }
#' @S3method predict bayesreg
#' @method predict bayesreg
#' @export
predict.bayesreg <- function(object, newdata, type = "linpred", bayes.avg = FALSE, sum.stat = "mean", CI = 95, ...)
{
  if (!inherits(object,"bayesreg")) stop("Not a valid Bayesreg object")

  # Error checking 
  if (sum.stat != "mean" && sum.stat != "median")
  {
    stop("The summary statistic must be either 'mean' or 'median'.")
  }
  if (type == "class" && object$model != "logistic")
  {
    stop("Class predictions are only available for logistic regressions.")
  }
  if (type != "linpred" && type != "prob" && type != "class" && type != "response")
  {
    stop("Type of prediction must be one of 'linpred', 'prob', 'class' or 'response'.")
  }
  if (CI <= 0 || CI >= 100)
  {
    stop("Credible interval level must be between 0 and 100 (exclusive).")
  }
  
  # Build the fully specified formula using the covariates that were fitted
  f <- stats::as.formula(paste("~",paste(attr(object$terms,"term.labels"),collapse="+")))
  
  # Extract the design matrix
  X = stats::model.matrix(f, data=newdata)
  X = as.matrix(X[,-1])
  n = nrow(X)
  p = ncol(X)

  # Get y-data if it has been passed and is not NA
  if (!any(names(newdata) == object$target.var) && type == "prob" && object$model != "logistic")
  {
    stop("You must provide a column of targets called '", object$target.var, "' to predict probabilities.")
  }
  
  if (any(names(newdata) == object$target.var) && type == "prob")
  {
    y <- as.matrix(newdata[object$target.var])
    if (any(is.na(y)) && type == "prob" && object$model != "logistic")
    {
      stop("Missing values in the target variable '", object$target.var, "' not allowed when predicting probabilities.")
    }
  }
  else {
    y <- NA
  }

  # Compute the linear predictor
  # If not averageing
  if (bayes.avg == F)
  {
    #browser()
    if (sum.stat == "median")
    {
      medBeta  = apply(object$beta,1,function(x) stats::quantile(x,c(0.5)))
      medBeta0 = stats::quantile(object$beta0,0.5)
      yhat = X %*% as.vector(medBeta) + as.numeric(medBeta0)
    }
    else
    {
      yhat = X %*% as.vector(object$mu.beta) + as.numeric(object$mu.beta0)
    }
  }
  # Otherwise we are averaging
  else
  {
    yhat = X %*% as.matrix(object$beta)
    yhat = t(as.matrix(apply(yhat,1,function(x) x + object$beta0)))
  }
  
  # Logistic reg -- class labels
  if (object$model == "logistic" && type == "class")
  {
    yhat = rowMeans(yhat)
    yhat[yhat>0] <- 2
    yhat[yhat<=0] <- 1
    yhat = factor(yhat, levels=c(1,2), labels=object$ylevels)

  # Logistic reg -- response prob
  } else if (object$model == "logistic" && (type == "prob" || type == "response") )
  {
    # Compute the response -- i.e., probability of Y=1
    eps = 1e-16
    yhat = (1 / (1+exp(-yhat)) + eps)/(1+2*eps)
  
    # If binary and type="prob", and y has been passed, compute probabilities for the specific y values
    if (!any(is.na(y)) && type == "prob")
    {
      yhat[y == object$ylevels[1],] <- 1 - yhat[y == object$ylevels[1],]
    }
  }
  
  # Continuous linear regression -- probability
  else if ( (object$model == "gaussian" || object$model == "laplace" || object$model == "t") && type == "prob" && !any(is.na(y)))
  {
    # Gaussian probabilities
    if (object$model == "gaussian")
    {
      if (bayes.avg == T)
      {
        yhat <- (as.matrix(apply(yhat, 2, function(x) (x - y)^2 )))
        yhat <- t(as.matrix(apply(yhat, 1, function(x) (1/2)*log(2*pi*object$sigma2) + x/2/object$sigma2)))
      }
      else
      {
        scale = as.numeric(object$mu.sigma2)
        yhat <- (1/2)*log(2*pi*scale) + (yhat - y)^2/2/scale
      }
    }

    # Laplace probabilities
    else if (object$model == "laplace")
    {
      if (bayes.avg == T)
      {
        yhat <- as.matrix(apply(yhat, 2, function(x) abs(x - y) ))
        scale <- as.matrix(sqrt(object$sigma2/2))
        yhat <- t(as.matrix(apply(yhat, 1, function(x) log(2*scale) + x/scale)))
      }
      else
      {
        scale <- sqrt(as.numeric(object$mu.sigma2)/2)
        yhat <- log(2*scale) + abs(yhat - y)/scale
      }
    }
    
    # Student-t probabilities
    else
    {
      nu = object$t.dof;
      if (bayes.avg == T)
      {
        yhat <- (as.matrix(apply(yhat, 2, function(x) x - y)))^2
        scale = as.matrix(object$sigma2)
        yhat <- t(as.matrix(apply(yhat, 1, function(x) (-lgamma((nu+1)/2) + lgamma(nu/2) + log(pi*nu*scale)/2) + (nu+1)/2*log(1 + 1/nu*x/scale) )))
      }
      else
      {
        yhat <- (-lgamma((nu+1)/2) + lgamma(nu/2) + log(pi*nu*as.numeric(object$mu.sigma2))/2) + (nu+1)/2*log(1 + 1/nu*(yhat-y)^2/as.numeric(object$mu.sigma2))
      }
    }
    
    yhat <- exp(-yhat)
  }
  
  # Count regression
  if ((object$model == "poisson" || object$model == "geometric") && type != "linpred")
  {
    # Response
    if (type == "response")
    {
      yhat <- exp(yhat)
    }
    
    # Otherwise, probabilities
    else
    {
      # Poisson
      if (object$model == "poisson")
      {
        if (bayes.avg == F)
        {
          yhat <- (exp(yhat)) - ( y*yhat ) + lgamma(y+1)
        }
        else
        {
          yhat = exp(yhat) - apply(yhat, 2, function(x) (x*y))
          yhat = apply(yhat, 2, function(x)(x+lgamma(y+1)))
        }
      }
      # Geometric
      if (object$model == "geometric")
      {
        eps = 1e-16
        if (bayes.avg == F)
        {
          geo.p = 1/(1+exp(yhat))
          geo.p = (geo.p+eps)/(1+2*eps)
          yhat  = -y*log(1-geo.p) - log(geo.p)
        }
        else
        {
          geo.p = 1/(1+exp(yhat))
          geo.p = (geo.p+eps)/(1+2*eps)
          
          yhat = apply(-log(1-geo.p), 2, function(x)(x*y))
          yhat = yhat - log(geo.p)
        }
      }
      
      yhat = exp(-yhat)
    }
  }

  # If not class labels and averaging, also compute SE's and CI's
  if (type != "class" && bayes.avg == T)
  {
    # Standard errors
    se = apply(yhat,1,stats::sd)
    
    # Credible intervals
    CI = round(CI,2)
    CI.lwr = (1 - CI/100)/2
    CI.upr = 1 - CI.lwr
    CI = apply(yhat,1,function(x) stats::quantile(x,c(CI.lwr,0.5,CI.upr)))
    
    # Store results
    r = matrix(0, n, 4)
    if (sum.stat == "median" && type != "prob")
    {
      r[,1] = as.matrix(CI[2,])
    }
    else
    {
      r[,1] = as.matrix(rowMeans(yhat))
    }
    r[,2] = as.matrix(se)
    r[,3] = as.matrix(CI[1,])
    r[,4] = as.matrix(CI[3,])
    colnames(r) <- c("pred","se(pred)",sprintf("CI %g%%", CI.lwr*100),sprintf("CI %g%%", CI.upr*100))
  }
  
  # Otherwise just return the class labels
  else
  {
    r = yhat
    if (object$model != "logistic")
    {
      colnames(r) <- "pred"
    }
  }
  
  return(r)
}


#' \code{summary} method for Bayesian regression models fitted using \code{\link{bayesreg}}.
#'
#' @title Summarization method for Bayesian penalised regression (\code{bayesreg}) models
#' @param object An object of class \code{"bayesreg"} created as a result of a call to \code{\link{bayesreg}}.
#' @param sort.rank logical; if \code{TRUE}, the variables in the summary will be sorted by their importance as determined by their rank estimated by 
#' the Bayesian feature ranking algorithm.
#' @param display.OR logical; if \code{TRUE}, the variables will be summarised in terms of their cross-sectional odds-ratios rather than their 
#' regression coefficients (logistic regression only).
#' @param CI numerical; the level of the credible interval reported in summary. Default is 95 (i.e., 95\% credible interval).
#' @param ... Further arguments passed to or from other methods.
#' @section Details:
#' The \code{summary} method computes a number of summary statistics and displays these for each variable in a table, along 
#' with suitable header information.
#' 
#' For continuous target variables, the header information includes a posterior estimate of the standard deviation of the random disturbances (errors), the \eqn{R^2} statistic
#' and the Widely applicable information criterion (WAIC) statistic. For logistic regression models, the header information includes the negative 
#' log-likelihood at the posterior mean of the regression coefficients, the pseudo \eqn{R^2} score and the WAIC statistic. For count
#' data (Poisson and geometric), the header information includes an estimate of the degree of overdispersion (observed variance divided by expected variance around the conditional mean, with a value < 1 indicating underdispersion),
#' the pseudo \eqn{R^2} score and the WAIC statistic.
#' 
#' The main table summarises properties of the coefficients for each of the variables. The first column is the variable name. The 
#' second and third columns are either the mean and standard error of the coefficients, or the median and standard error of the 
#' cross-sectional odds-ratios if \code{display.OR=TRUE}. 
#' 
#' The fourth and fifth columns are the end-points of the credible intervals of the coefficients (odds-ratios). The sixth column displays the 
#' posterior \eqn{t}-statistic, calculated as the ratio of the posterior mean on the posterior standard deviation for the coefficient. 
#' The seventh column is the importance rank assigned to the variable by the Bayesian feature ranking algorithm. 
#' 
#' In between the seventh and eighth columns are up to two asterisks indicating significance; a variable scores a first asterisk if 
#' the 75\% credible interval does not include zero, and scores a second asterisk if the 95\% credible interval does not include zero. The 
#' final column gives an estimate of the effective sample size for the variable, ranging from 0 to n.samples, which indicates the 
#' effective number of i.i.d draws from the posterior (if we could do this instead of using MCMC) represented by the samples
#' we have drawn. This quantity is computed using the algorithm presented in the Stan Bayesian sampling package documentation.
#' @return Returns an object with the following fields:
#'   
#' \item{log.l}{The log-likelihood of the model at the posterior mean estimates of the regression coefficients.}
#' \item{waic}{The Widely Applicable Information Criterion (WAIC) score of the model.}
#' \item{waic.dof}{The effective degrees-of-freedom of the model, as estimated by the WAIC.}
#' \item{r2}{For non-binary data, the R^2 statistic.}
#' \item{sd.error}{For non-binary data, the estimated standard deviation of the errors.}
#' \item{p.r2}{For binary data, the pseudo-R^2 statistic.}
#' \item{mu.coef}{The posterior means of the regression coefficients.}
#' \item{se.coef}{The posterior standard deviations of the regression coefficients.}
#' \item{CI.coef}{The posterior credible interval for the regression coefficients, at the level specified (default: 95\%).}
#' \item{med.OR}{For binary data, the posterior median of the cross-sectional odds-ratios.}
#' \item{se.OR}{For binary data, the posterior standard deviation of the cross-sectional odds-ratios.}
#' \item{CI.OR}{For binary data, the posterior credible interval for the cross-sectional odds-ratios.}
#' \item{t.stat}{The posterior t-statistic for the coefficients.}
#' \item{n.stars}{The significance level for the variable (see above).}
#' \item{rank}{The variable importance rank as estimated by the Bayesian feature ranking algorithm (see above).}
#' \item{ESS}{The effective sample size for the variable.}
#' \item{log.l0}{For binary data, the log-likelihood of the null model (i.e., with only an intercept).}
#' @seealso The model fitting function \code{\link{bayesreg}} and prediction function \code{\link{predict.bayesreg}}.
#' @examples
#' 
#' X = matrix(rnorm(100*20),100,20)
#' b = matrix(0,20,1)
#' b[1:9] = c(0,0,0,0,5,4,3,2,1)
#' y = X %*% b + rnorm(100, 0, 1)
#' df <- data.frame(X,y)
#' 
#' rv.hs <- bayesreg(y~.,df,prior="hs")       # Horseshoe regression
#' 
#' # Summarise without sorting by variable rank
#' rv.hs.s <- summary(rv.hs)
#' 
#' # Summarise sorting by variable rank and provide 75% credible intervals
#' rv.hs.s <- summary(rv.hs, sort.rank = TRUE, CI=75)
#'
#' @S3method summary bayesreg
#' @method summary bayesreg
#' @export
summary.bayesreg <- function(object, sort.rank = FALSE, display.OR = FALSE, CI = 95, ...)
{
    VERSION = '1.2'
    
    # Credible interval must be between 1 and 99
    CI = round(CI)
    if (CI < 1 || CI > 99)
    {
      stop("Credible interval level must be between 1 and 99.")
    }
    CI.low  = (1 - CI/100)/2
    CI.high = 1- CI.low
  
    if (!inherits(object,"bayesreg")) stop("Not a valid Bayesreg object.")
  
    printf <- function(...) cat(sprintf(...))
    repchar <- function(c, n) paste(rep(c,n),collapse="")

    n   = object$n
    px  = object$p
    
    PerSD = F

    rv = list()

    # Error checking
    if (display.OR == T && object$model != "logistic")
    {
      stop("Can only display cross-sectional odds-ratios for logistic models.")
    }
        
    # Banner
    printf('==========================================================================================\n');
    printf('|                   Bayesian Penalised Regression Estimation ver. %s                    |\n', VERSION);
    printf('|                     (c) Enes Makalic, Daniel F Schmidt. 2016-2021                      |\n');
    printf('==========================================================================================\n');
    
    # ===================================================================
    # Table symbols
    chline = '-'
    cvline = '|'
    cTT    = '+'
    cplus  = '+'
    bTT    = '+'
    
    # ===================================================================
    # Find length of longest variable name
    varnames = c(labels(object$mu.beta)[[1]], "_cons")
    maxlen   = 12
    nvars    = length(varnames)
    for (i in 1:nvars)
    {
      if (nchar(varnames[i]) > maxlen)
      {
        maxlen = nchar(varnames[i])
      }
    }
    
    # ===================================================================
    # Display pre-table stuff
    #printf('%s%s%s\n', repchar('=', maxlen+1), '=', repchar('=', 76))
    #printf('\n')
    
    if (object$model != 'logistic')
    {
      model = 'linear'
    } else {
      model = 'logistic'
    }
    
    if (object$model == "gaussian")
    {
      distr = "Gaussian"
    } else if (object$model == "laplace")
    {
      distr = "Laplace"
    }
    else if (object$model == "t")
    {
      distr = paste0("Student-t (DOF = ",object$t.dof,")")
    }
    else if (object$model == "logistic")
    {
      distr = "logistic"
    }
    else if (object$model == "poisson")
    {
      distr = "Poisson"
    }
    else if (object$model == "geometric")
    {
      distr = "geometric"
    }
    
    if (object$prior == 'rr' || object$prior == 'ridge') {
      prior = 'ridge'
    } else if (object$prior == 'lasso') {      
      prior = 'lasso'
    } else if (object$prior == 'hs' || object$prior == 'horseshoe') {
      prior = 'horseshoe'
    } else if (object$prior == 'hs+' || object$prior == 'horseshoe+') {
      prior = 'horseshoe+'
    }

    str = sprintf('Bayesian %s %s regression', distr, prior)
    printf('%-64sNumber of obs   = %8d\n', str, n)
    printf('%-64sNumber of vars  = %8.0f\n', '', px);
    
    if (object$model == 'gaussian' || object$model == 'laplace' || object$model == 't')
    {
      s2 = mean(object$sigma2)
      if (object$model == 't' && object$t.dof > 2)
      {
        s2 = object$t.dof/(object$t.dof - 2) * s2
      }
      else if (object$model == 't' && object$t.dof <= 2)
      {
        s2 = NA
      }

      str = sprintf('MCMC Samples   = %6.0f', object$n.samples);
      if (!is.na(s2))
      {
        printf('%-64sstd(Error)      = %8.5g\n', str, sqrt(s2));
      }
      else {
        printf('%-64sstd(Error)      =        -\n', str);
      }
          
      str = sprintf('MCMC Burnin    = %6.0f', object$burnin);
      printf('%-64sR-squared       = %8.4f\n', str, object$r2);    
      str = sprintf('MCMC Thinning  = %6.0f', object$thin);
      printf('%-64sWAIC            = %8.5g\n', str, object$waic);
      
      rv$sd.error  = sqrt(s2)
      rv$r2        = object$r2
      rv$waic      = object$waic
      rv$waic.dof  = object$waic.dof
      rv$log.l     = object$log.l
    }
    else if (object$model == 'logistic')
    {
      log.l = object$log.l
      log.l0 = object$log.l0
      r2 = 1 - log.l / log.l0
      
      str = sprintf('MCMC Samples   = %6.0f', object$n.samples);
      printf('%-64sLog. Likelihood = %8.5g\n', str, object$log.l);
      str = sprintf('MCMC Burnin    = %6.0f', object$burnin);
      printf('%-64sPseudo R2       = %8.4f\n', str, r2);    
      str = sprintf('MCMC Thinning  = %6.0f', object$thin);
      printf('%-64sWAIC            = %8.5g\n', str, object$waic);
      
      rv$log.l     = log.l
      rv$p.r2      = r2
      rv$waic      = object$waic
      rv$waic.dof  = object$waic.dof
    }
    else if (object$model == 'poisson' || object$model == 'geometric')
    {
      log.l = object$log.l
      log.l0 = object$log.l0
      r2 = 1 - log.l / log.l0
      
      str = sprintf('MCMC Samples   = %6.0f', object$n.samples);
      printf('%-64sOverdispersion  = %8.5g\n', str, object$over.dispersion);
      str = sprintf('MCMC Burnin    = %6.0f', object$burnin);
      printf('%-64sPseudo R2       = %8.4f\n', str, r2);    
      str = sprintf('MCMC Thinning  = %6.0f', object$thin);
      printf('%-64sWAIC            = %8.5g\n', str, object$waic);
      
      rv$log.l     = log.l
      rv$p.r2      = r2
      rv$waic      = object$waic
      rv$waic.dof  = object$waic.dof      
    }
    printf('\n')
    
    # ===================================================================
    # Table Header
    fmtstr = sprintf('%%%ds', maxlen);
    printf('%s%s%s\n', repchar(chline, maxlen+1), cTT, repchar(chline, 77))
    tmpstr = sprintf(fmtstr, 'Parameter');
    if (model == 'linear')
    {
      printf('%s %s  %10s %10s    [%d%% Cred. Interval] %10s %7s %10s\n', tmpstr, cvline, 'mean(Coef)', 'std(Coef)', CI, 'tStat', 'Rank', 'ESS');
    } else if (model == 'logistic' && display.OR == F) {
      printf('%s %s  %10s %10s    [%d%% Cred. Interval] %10s %7s %10s\n', tmpstr, cvline, 'med(Coef)', 'std(OR)', CI, 'tStat', 'Rank', 'ESS');
    } else if (model == 'logistic' && display.OR == T) {
      printf('%s %s  %10s %10s    [%d%% Cred. Interval] %10s %7s %10s\n', tmpstr, cvline, 'med(OR)', 'std(OR)', CI, 'tStat', 'Rank', 'ESS');
    }
    printf('%s%s%s\n', repchar(chline, maxlen+1), cTT, repchar(chline, 77));

    # ===================================================================
    if (PerSD) {
      beta0 = object$beta0
      beta  = object$beta
      
      object$beta0   = beta0 + object$std.X$mean.X %*% object$beta
      object$beta    = apply(t(object$beta), 1, function(x)(x * object$std.X$std.X/sqrt(n)))    
      object$mu.beta = object$mu.beta * t(as.matrix(object$std.X$std.X)) / sqrt(n)
    }

    # ===================================================================
    # Return values
    rv$mu.coef   = matrix(0,px+1,1, dimnames = list(varnames))
    rv$se.coef   = matrix(0,px+1,1, dimnames = list(varnames))
    rv$CI.coef   = matrix(0,px+1,2, dimnames = list(varnames))
    
    if (model == "logistic")
    {
      rv$med.OR  = matrix(0,px+1,1, dimnames = list(varnames))
      rv$se.OR   = matrix(0,px+1,1, dimnames = list(varnames))
      rv$CI.OR   = matrix(0,px+1,2, dimnames = list(varnames))
    }

    rv$t.stat    = matrix(0,px+1,1, dimnames = list(varnames))
    rv$n.stars   = matrix(0,px+1,1, dimnames = list(varnames))
    rv$ESS       = matrix(0,px+1,1, dimnames = list(varnames))
    rv$rank     = matrix(0,px+1,1, dimnames = list(varnames))

    # Variable information
    rv$rank[1:(px+1)] = as.matrix(object$var.ranks)

    # If sorting by BFR ranks
    if (sort.rank == T)
    {
      O = sort(object$var.ranks, index.return = T)
      indices = O$ix
      indices[px+1] = px+1
    }    
    # Else sorted by the order they were passed in
    else {
      indices = (1:(px+1))
    }
    
    for (i in 1:(px+1))
    {
      k = indices[i]
      kappa = NA
      
      # Regression variable
      if (k <= px)
      {
        s = object$beta[k,]
        mu = object$mu.beta[k]

        # Calculate shrinkage proportion, if possible
        kappa = object$t.stat[k]
      }
      
      # Intercept
      else if (k == (px+1))
      {
        s = object$beta0
        mu = mean(s)
      }
      
      # Compute credible intervals/standard errors for beta's
      std_err = stats::sd(s)
      qlin = stats::quantile(s, c(CI.low, CI.high))
      qlog = stats::quantile(exp(s), c(CI.low, CI.high))
      q = qlin

      rv$mu.coef[k]  = mu
      rv$se.coef[k]  = std_err
      rv$CI.coef[k,] = c(qlin[1],qlin[2])

      # Compute credible intervals/standard errors for OR's
      if (model == 'logistic')
      {
        med_OR = stats::median(exp(s))
        std_err_OR = (qlog[2]-qlog[1])/2/1.96
        
        rv$med.OR[k] = med_OR
        rv$se.OR[k]  = std_err_OR
        rv$CI.OR[k,] = c(qlog[1],qlog[2])

        # If display ORs, use these instead
        if (display.OR)
        {
          mu = med_OR
          std_err = std_err_OR
          q = qlog
        }
        # Otherwise use posterior medians of coefficients for stability
        else
        {
          mu = stats::median(s)
        }
      }
      rv$t.stat[k] = kappa
      
      # Display results
      tmpstr = sprintf(fmtstr, varnames[k])
      if (is.na(kappa))
        t.stat = '         .'
      else t.stat = sprintf('%10.3f', kappa)
      
      if (is.na(object$var.ranks[k]))
        rank = '      .'
      else
        rank = sprintf('%7d', object$var.ranks[k])

      printf('%s %s %11.5f %10.5f   %10.5f %10.5f %s %s ', tmpstr, cvline, mu, std_err, q[1], q[2], t.stat, rank);

      # Model selection scores
      qlin = stats::quantile(s, c(0.025, 0.125, 0.875, 0.975))
      qlog = stats::quantile(exp(s), c(0.025, 0.125, 0.875, 0.975))

      # Test if 75% CI includes 0
      if ( k <= px && ( (qlin[2] > 0 && qlin[3] > 0) || (qlin[2] < 0 && qlin[3] < 0) ) )    
      {
        printf('*')
        rv$n.stars[k] = rv$n.stars[k] + 1
      }
      else 
        printf(' ')
      
      # Test if 95% CI includes 0
      if ( k <= px && ( (qlin[1] > 0 && qlin[4] > 0) || (qlin[1] < 0 && qlin[4] < 0) ) )    
      {
        printf('*')
        rv$n.stars[k] = rv$n.stars[k] + 1
      }
      else
        printf(' ')

      # Display ESS-frac
      if(k > px)
        printf('%8s', '.')
      else
        printf('%8d', object$ess[k])

      rv$ESS[k] = object$ess[k]
      
      printf('\n');
    }
    
    printf('%s%s%s\n\n', repchar(chline, maxlen+1), cTT, repchar(chline, 77));
    
    if (model == 'logistic')
    {
      rv$log.l0 = object$log.l0
    }
    
    invisible(rv)
}


# ============================================================================================================================
# function to standardise columns of X to have mean zero and unit length
bayesreg.standardise <- function(X)
{
  n = nrow(X)
  p = ncol(X)
  
  # 
  r       = list()
  r$X     = X
  if (p > 1)
  {
    r$mean.X = colMeans(X)
  } else
  {
    r$mean.X = mean(X)
  }
  r$std.X  = t(apply(X,2,stats::sd)) * sqrt(n-1)
  
  # Perform the standardisation
  if (p == 1)
  {
    r$X <- as.matrix(apply(X,1,function(x)(x - r$mean.X)))
    r$X <- as.matrix(apply(r$X,1,function(x)(x / r$std.X)))
  } else
  {
    r$X <- t(as.matrix(apply(X,1,function(x)(x - r$mean.X))))
    r$X <- t(as.matrix(apply(r$X,1,function(x)(x / r$std.X))))
  }
  
  return(r)
}


# ============================================================================================================================
# Sample the regression coefficients
bayesreg.sample_beta <- function(X, z, mvnrue, b0, sigma2, tau2, lambda2, omega2, XtX, Xty, model, PrecomputedXtX)
{
  alpha  = (z - b0)
  Lambda = sigma2 * tau2 * lambda2
  sigma  = sqrt(sigma2)
  
  # Use Rue's algorithm
  if (mvnrue)
  {
    # If XtX is not precomputed
    if (!PrecomputedXtX)
    {
      #omega = sqrt(omega2)
      #X0    = apply(X,2,function(x)(x/omega))
      #bs    = bayesreg.fastmvg2_rue(X0/sigma, alpha/sigma/omega, Lambda)

      omega = sqrt(omega2)*sigma
      X0 = X
      # Loop is faster than apply() -- why??
      for (i in 1:length(lambda2))
      {
        X0[,i] = X[,i]/omega
      }

      bs    = bayesreg.fastmvg2_rue.nongaussian(X0, alpha, Lambda, sigma2, omega)
    }
    
    # XtX is precomputed (Gaussian only)
    else {
      #bs    = bayesreg.fastmvg2_rue(X/sigma, alpha/sigma, Lambda, XtX/sigma2)
      bs    = bayesreg.fastmvg2_rue.gaussian(X, alpha, Lambda, XtX, Xty, sigma2, PrecomputedXtX=T)
      #bs    = bayesreg.fastmvg2_rue.gaussian.2(X, alpha, Lambda, XtX, sigma2, PrecomputedXtX=T)
    }
  }
  
  # Else use Bhat. algorithm
  else
  {
    omega = sqrt(omega2)*sigma
    
    # Non-Gaussian (heteroskedastic Gaussian ...)
    if (model != 'gaussian')
    {
      #omega = sqrt(omega2)
      #X0    = apply(X,2,function(x)(x/omega))
      #bs    = bayesreg.fastmvg_bhat(X0/sigma, alpha/sigma/omega, Lambda)
      
      # Loop is faster than apply ...
      X0    = X
      for (i in 1:length(lambda2))
      {
        X0[,i] = X[,i]/omega
      }
    }
    # Gaussian 
    else
    {
      X0 = X/sigma
    }
    
    # Generate a sample
    bs    = bayesreg.fastmvg_bhat(X0, alpha/omega, Lambda)
  }
  
  return(bs)
}


# ============================================================================================================================
# function to generate multivariate normal random variates using Rue's algorithm for non-Gaussian noise
bayesreg.fastmvg2_rue.nongaussian <- function(Phi, alpha, d, sigma2, omega)
{
  Phi   = as.matrix(Phi)
  alpha = as.matrix(alpha)
  r     = list()

  p     = ncol(Phi)
  Dinv  = diag(as.vector(1/d), nrow = length(d))

  PtP   = t(Phi)%*%Phi
  
  L     = chol(PtP + Dinv)
  v     = forwardsolve(t(L), (crossprod(Phi,alpha/omega)))
  r$m   = backsolve(L, v)
  w     = backsolve(L, stats::rnorm(p,0,1))
  
  r$x   = r$m + w
  return(r)
}


# ============================================================================================================================
# function to generate multivariate normal random variates using Rue's algorithm for Gaussian noise
bayesreg.fastmvg2_rue.gaussian <- function(Phi, alpha, d, PtP = NA, Ptalpha = NA, sigma2, PrecomputedXtX=F)
{
  Phi   = as.matrix(Phi)
  alpha = as.matrix(alpha)
  r     = list()
  
  # If PtP not precomputed
  if (!PrecomputedXtX)
  {
    PtP = t(Phi) %*% Phi
  }

  p     = ncol(Phi)
  Dinv  = diag(as.vector(1/d), nrow = length(d))

  L     = chol(PtP/sigma2 + Dinv)
  
  v     = forwardsolve(t(L), (Ptalpha/sigma2))
  r$m   = backsolve(L, v)
  w     = backsolve(L, stats::rnorm(p,0,1))
  
  r$x   = r$m + w
  return(r)
}

# ============================================================================================================================
# function to generate multivariate normal random variates using Bhat. algorithm
bayesreg.fastmvg_bhat <- function(Phi, alpha, d)
{
  #d     = as.matrix(d)
  p     = ncol(Phi)
  n     = nrow(Phi)
  r     = list()
  
  u     = as.matrix(stats::rnorm(p,0,1)) * sqrt(d)
  delta = as.matrix(stats::rnorm(n,0,1))
  
  v     = Phi %*% u + delta
  #Dpt   = (apply(Phi, 1, function(x)(x*d)))
  #
  #Dpt = Phi
  #for (i in 1:length(alpha))
  #{
  #  Dpt[i,] = Dpt[i,]*d
  #}
  #Dpt = t(Dpt)
    
  Dpt   = Phi
  for (i in 1:length(d))
  {
    Dpt[,i] = Dpt[,i]*d[i]
  }
  Dpt = t(Dpt)
  
  W     = Phi %*% Dpt + diag(1,n)

  #w     = solve(W,(alpha-v))

  L     = chol(W)
  vv    = forwardsolve(t(L), (alpha-v))
  w     = backsolve(L, vv)

  r$x   = u + Dpt %*% w
  r$m   = r$x
  
  return(r)
}

# ============================================================================================================================
# rinvg
bayesreg.rinvg <- function(mu, lambda)
{
  lambda = 1/lambda
  p = length(mu)
  
  V = stats::rnorm(p,mean=0,sd=1)^2
  out = mu + 1/2*mu/lambda * (mu*V - sqrt(4*mu*lambda*V + mu^2*V^2))
  out[out<1e-16] = 1e-16
  z = stats::runif(p)
  
  l = z >= mu/(mu+out)
  out[l] = mu[l]^2 / out[l]
  
  return(out)
}

# ============================================================================================================================
# compute the negative log-likelihood
bayesreg.linregnlike <- function(error, X, y, b, b0, s2, t.dof = NA)
{
  n = nrow(y)

  y = as.matrix(y)
  X = as.matrix(X)
  b = as.matrix(b)
  b0 = as.numeric(b0)
  
  e = (y - X %*% as.matrix(b) - as.numeric(b0))
  
  if (error == 'gaussian')
  {
    negll = n/2*log(2*pi*s2) + 1/2/s2*t(e) %*% e
  }
  else if (error == 'laplace')
  {
    scale <- sqrt(as.numeric(s2)/2)
    negll = n*log(2*scale) + sum(abs(e))/scale
  }
  else if (error == 't')
  {
    nu = t.dof;
    negll = n*(-lgamma((nu+1)/2) + lgamma(nu/2) + log(pi*nu*s2)/2) + (nu+1)/2*sum(log(1 + 1/nu*e^2/as.numeric(s2)))
  }
  
  return(negll)
}

# ============================================================================================================================
# compute the negative log-likelihood for logistic regression
bayesreg.logregnlike <- function(X, y, b, b0)
{
  y = as.matrix(y)
  X = as.matrix(X)
  b = as.matrix(b)
  b0 = as.numeric(b0)
  
  eps = exp(-36)
  lowerBnd = log(eps)
  upperBnd = -lowerBnd
  muLims = c(eps, 1-eps)
  
  eta = as.numeric(b0) + X %*% as.matrix(b)
  eta[eta < lowerBnd] = lowerBnd
  eta[eta > upperBnd] = upperBnd
  
  mu = 1 / (1 + exp(-eta))
  
  mu[mu < eps] = eps
  mu[mu > (1-eps)] = (1-eps)
  
  negll = -sum(y*log(mu)) -
    sum((1.0 - y)*log(1.0 - mu))
  
  return(negll)
}

# ============================================================================================================================
# compute the negative log-likelihood for count regression
bayesreg.countregnlike <- function(model, X, y, b, b0)
{
  n = nrow(y)
  
  y = as.matrix(y)
  X = as.matrix(X)
  b = as.matrix(b)
  b0 = as.numeric(b0)
  
  eta = X %*% as.matrix(b) + as.numeric(b0)

  if (model == 'poisson')
  {
    negll = sum(-y*eta + exp(eta) + lgamma(y+1))
  }
  else if (model == 'geometric')
  {
    mu = exp(eta)
    geo.p = 1/(1+exp(eta))
    
    eps = exp(-36)
    geo.p[geo.p < eps] = eps
    geo.p[geo.p > (1-eps)] = (1-eps)    
    
    negll = sum(-y*log(1-geo.p) - log(geo.p))
  }

  return(negll)
}

# ============================================================================================================================
# Compute individual likelihoods for linear regression using precomputed predictor
bayesreg.linregnlike_e <- function(error, e, s2, t.dof = NA)
{
  n = nrow(e)

  if (error == 'gaussian')
  {
    negll = 1/2*log(2*pi*s2) + 1/2/s2*e^2
  }
  else if (error == 'laplace')
  {
    scale <- sqrt(as.numeric(s2)/2)
    negll = 1*log(2*scale) + abs(e)/scale
  }
  else if (error == 't')
  {
    nu = t.dof;
    negll = 1*(-lgamma((nu+1)/2) + lgamma(nu/2) + log(pi*nu*s2)/2) + (nu+1)/2*(log(1 + 1/nu*e^2/as.numeric(s2)))
  }

  rv = list(negll = negll, prob = exp(-negll))
  
  return(rv)
}

# ============================================================================================================================
# Compute individual likelihoods for logistic regression using precomputed predictor
bayesreg.logregnlike_eta <- function(y, eta)
{
  y = as.matrix(y)
  b = as.matrix(eta)

  eps = exp(-36)
  lowerBnd = log(eps)
  upperBnd = -lowerBnd
  muLims = c(eps, 1-eps)
  
  # Constrain the linear predictor
  eta[eta < lowerBnd] = lowerBnd
  eta[eta > upperBnd] = upperBnd
  
  mu = 1 / (1 + exp(-eta))
  
  # Constrain probabilities 
  mu[mu < eps] = eps
  mu[mu > (1-eps)] = (1-eps)
  
  # Return quantities
  negll = -y*log(mu) - (1.0-y)*log(1.0-mu)
  
  rv = list(negll = negll, prob = exp(-negll))

  return(rv)
}

# ============================================================================================================================
# Compute individual likelihoods for count regression using precomputed predictor
bayesreg.countregnlike_eta <- function(model, y, eta)
{
  if (model == 'poisson')
  {
    negll = -y*eta + exp(eta) + lgamma(y+1)
  }
  else if (model == 'geometric')
  {
    mu = exp(eta)
    geo.p = 1/(1+exp(eta))

    eps = exp(-36)
    geo.p[geo.p < eps] = eps
    geo.p[geo.p > (1-eps)] = (1-eps)    

    negll = -y*log(1-geo.p) - log(geo.p)
  }
  
  rv = list(negll = negll, prob = exp(-negll))
}

# ============================================================================================================================
# Simple gradient descent fitting of generalized linear models
bayesreg.fit.GLM.gd <- function(X, y, model, xi, tau2, max.iter = 5e2)
{
  nx = nrow(X)
  px = ncol(X)
  theta = matrix(data=0, nrow = px+1, ncol = 1)
  
  # Precompute statistics
  Xty = crossprod(X,y)
  
  # Learning rate
  kappa = 1
  
  # Error checking
  # ...  
  
  # Any special initialisation
  if (model == 'gaussian')
  {
    theta[px+1] = mean(y)
  }
  
  # Optimise
  rv.L = bayesreg.grad.L(y, X, theta, model, Xty, xi, tau2)
  for (i in 1:max.iter)
  {
    # Update estimates
    kappa.vec = matrix(kappa,px+1,1)
    kappa.vec[px+1] = kappa.vec[px+1]/sqrt(nx)
    theta.new = theta - kappa.vec*rv.L$grad
    
    # Have we improved?
    rv.L.new = bayesreg.grad.L(y, X, theta.new, model, Xty, xi, tau2)
    if (rv.L.new$L < rv.L$L)
    {
      theta = theta.new
      rv.L  = rv.L.new
    } 
    else 
    {
      # If not, halve the learning rate
      kappa = kappa/2
    }
  }
  
  # Return
  rv = list()
  rv$b       = theta[1:px]
  rv$b0      = theta[px+1]
  rv$grad.b  = rv.L$grad[1:px]
  rv$grad.b0 = rv.L$grad[px+1]
  rv$L       = rv.L$L
  
  rv
}

# ============================================================================================================================
# Simple likelihood/gradient function for gradient descent
bayesreg.grad.L <- function(y, X, theta, model, Xty, xi, tau2)
{
  px = length(theta)-1
  rv = list()
  
  # Poisson regression
  if (model == 'poisson')
  {
    # Form the linear predictor
    eta = X %*% theta[1:px] + theta[px+1]
    eta = pmin(eta, 500)
    mu  = as.vector(exp(eta))
    
    # Poisson likelihood (up to constants)
    rv$L = sum(mu) - sum( y*eta ) + sum(theta[1:px]^2)/2/tau2
    
    # Poisson gradient
    rv$grad = matrix(0, px+1, 1)
    rv$grad[1:px] = crossprod(X,mu) - Xty + theta[1:px]/tau2;
    rv$grad[px+1] = sum(mu) - sum(y)
  }
  
  else if (model == 'geometric')
  {
    r = xi[1]
    
    #browser()
    
    # Form the linear predictor
    eta = X %*% theta[1:px] + theta[px+1]
    eta = pmin(eta, 500)
    mu  = as.vector(exp(eta))
    
    # Geometric likelihood (up to constants)
    rv$L = -sum(eta*y) + sum(log(mu+r)*(y+1)) + sum(theta[1:px]^2)/2/tau2
    
    # Geometric gradient
    rv$grad = matrix(0, px+1, 1)
    c = as.vector((mu*(y+1)/(mu+r)))
    rv$grad[1:px] = crossprod(X,c) - Xty + theta[1:px]/tau2
    rv$grad[px+1] = sum(c) - sum(y)
  }
  
  rv
}

# ============================================================================================================================
# Gradient and likelihoods for GLM models (for use with mgrad tools)
bayesreg.mgrad.L <- function(y, X, theta, eta, model, xi, Xty)
{
  rv = list()
  px = ncol(X)
  
  # Form the linear predictor if needed
  if (is.null(eta))
  {
    rv$eta = as.vector(X %*% theta[1:px] + theta[px+1])
  }  
  else
  {
    rv$eta = eta
  }
  
  # Poisson regression
  if (model == 'poisson')
  {
    # Form the linear predictor
    #rv$eta = pmin(rv$eta, 500);
    rv$eta[rv$eta>500] = 500
    mu  = exp(rv$eta)
  
    # Poisson likelihood (up to constants)
    rv$L = sum(mu) - sum( y*rv$eta )
  
    # Poisson gradient
    rv$grad = matrix(0, px+1, 1)
    rv$grad[1:px] = crossprod(X,mu) - Xty
    rv$grad[px+1]   = sum(mu) - sum(y)
    
    rv$H.b0 = sum(mu)
  }
  
  # Geometric regression
  if (model == "geometric")
  {
    r = xi[1]
    
    rv$eta[rv$eta>500] = 500
    mu  = exp(rv$eta)
    
    # Geometric likelihood (up to constants)
    rv$L = -sum(rv$eta*y) + sum(log(mu+r)*(y+1))
    
    # Geometric gradient
    rv$grad = matrix(0, px+1, 1)
    c = as.vector((mu*(y+1)/(mu+r)))
    rv$grad[1:px] = crossprod(X,c) - Xty
    rv$grad[px+1] = sum(c) - sum(y)
    
    rv$H.b0 = sum(mu/(mu+1))
  }
  
  #
  rv$extra.stats = NULL
  
  rv
}

# ============================================================================================================================
# Create a tuning structure for adaptive step-size tuning
bayesreg.mh.initialise <- function(window, delta.max, delta.min, burnin, display)
{
  tune = list()
  
  # Step-size setup
  tune$M      = 0
  tune$window = window
  
  tune$delta.max = delta.max
  tune$delta.min = delta.min
  
  # Start in phase 1
  tune$iter            = 0
  tune$burnin          = burnin
  tune$W.phase         = 1
  tune$W.burnin        = 0
  tune$nburnin.windows = floor(burnin/window)
  tune$m.window        = rep(0, tune$nburnin.windows)
  tune$n.window        = rep(0, tune$nburnin.windows, 1)
  tune$delta.window    = rep(0, tune$nburnin.windows, 1)
  tune$display         = display
  
  tune$phase.cnt       = c(0,0,0)
  
  tune$M               = 0
  tune$D               = 0
  
  tune$b.tune          = NULL
  
  # Start at the maximum delta (phase 1)
  tune$delta           = delta.max
  
  tune
}

# ============================================================================================================================
# Sample the beta's using the mGrad algorithm
bayesreg.mgrad.sample.beta <- function(b, b0, L.b, grad.b, D, delta, eta, H.b0, y, X, sigma2, Xty, model, extra.model.params, extra.stats)
{
  px = ncol(X)

  # Get quantities need for proposal
  rv.prop = bayesreg.mgrad.update.proposal(px, D, delta)
  
  # Generate proposal from marginal proposal distribution    
  #bnew = normrnd( mGrad_prop_2.*((2/delta)*(b - (delta/2)*grad_b/sigma2)), sqrt((2/delta)*mGrad_prop_1+mGrad_prop_2) );
  b.new = rnorm( n=px, mean=rv.prop$prop.2*((2/delta)*(b - (delta/2)*grad.b/sigma2)), sd=sqrt((2/delta)*rv.prop$prop.1+rv.prop$prop.2) );
  
  # Accept/reject?
  extra.stats.new = NULL
  if (model == 'poisson')
  {
    rv.mh.new = bayesreg.mgrad.L(y, X, c(b.new,b0), NULL, 'poisson', extra.model.params, Xty)
  }
  if (model == 'geometric')
  {
    rv.mh.new = bayesreg.mgrad.L(y, X, c(b.new,b0), NULL, 'geometric', extra.model.params, Xty)
  }
  grad.b.new = rv.mh.new$grad[1:px]
  
  h1 = bayesreg.mgrad.hfunc(b, b.new, -grad.b.new/sigma2, rv.prop$prop.2, delta)
  h2 = bayesreg.mgrad.hfunc(b.new, b, -grad.b/sigma2, rv.prop$prop.2, delta);
  
  mhprob = min(exp(-rv.mh.new$L/sigma2 - -L.b/sigma2 + h1 - h2), 1)

  #if (is.na(mhprob) || is.nan(mhprob))
  #{
  #  mhprob
  #}
  
  # Check for acceptance  
  rv = list()
  rv$accepted = F
  if (runif(1) < mhprob && !any(is.infinite(b.new)))
  {
    rv$accepted = T
    
    rv$b = b.new;
    rv$b0 = b0
    rv$L.b = rv.mh.new$L
    rv$grad.b = grad.b.new
    rv$eta = rv.mh.new$eta
    rv$H.b0 = rv.mh.new$H.b0
    rv$extra.stats = extra.stats.new
  }
  # If reject, stay where we are
  else
  {
    rv$b = b
    rv$b0 = b0
    rv$L.b = L.b
    rv$grad.b = grad.b
    rv$eta = eta
    rv$H.b0 = H.b0
    rv$extra.stats = extra.stats
  }
  
  rv
}

# ============================================================================================================================
# Update the quantities used for MH proposal
bayesreg.mgrad.update.proposal <- function(px, Lambda, delta)
{
  rv = list()
  
  # Update proposal information
  rv$prop.2 = (1/(1/Lambda + (2/delta)))
  rv$prop.1 = rv$prop.2^2
  
  rv
}

# ============================================================================================================================
# 'h'-function for mGrad MH acceptance probability
bayesreg.mgrad.hfunc <- function(x, y, grad.y, v, delta)
{
  h = t((x - (2/delta)*v*(y + (delta/4)*grad.y))*(1/((2/delta)*v+1))) %*% grad.y
}

# ============================================================================================================================
# Tune the MH sampler step-size
bayesreg.mgrad.tune <- function(tune)
{
  tune$iter = tune$iter + 1

  DELTA.MAX = exp(40)
  DELTA.MIN = exp(-40)
  NUM.PHASE.1 = 15
  
  # Perform a tuning step, if necessary
  if ( (tune$iter %% tune$window) == 0)
  {
    #W_burnin, W_phase, delta, delta_min, delta_max, mc, nc, delta_window, m_window, n_window)

    # Store the measured acceptance probability
    tune$W.burnin                    = tune$W.burnin + 1;
    tune$delta.window[tune$W.burnin] = log(tune$delta);
    tune$m.window[tune$W.burnin]     = tune$M+1
    tune$n.window[tune$W.burnin]     = tune$D+2
    
    # If in phase 1, we are exploring the space uniformly
    if (tune$W.phase == 1)
    {
      tune$phase.cnt[1] = tune$phase.cnt[1]+1;
      #d = linspace(log(DELTA_MIN),log(DELTA_MAX),NUM_PHASE_1);
      d = seq(log(DELTA.MIN), log(DELTA.MAX), length.out = NUM.PHASE.1)
      
      tune$delta = exp(d[tune$phase.cnt[1]]);
      
      if (tune$delta < tune$delta.min)
      {
        tune$delta.min = tune$delta
      }
      if (tune$delta > tune$delta.max)
      {
        tune$delta.max = tune$delta
      }
      
      # If we have exhausted 
      if (tune$phase.cnt[1] == NUM.PHASE.1)
      {
        tune$W.phase = 2
      }
    }    
    # Else in phase 2, we are probing randomly guided by model
    else {
      tune$phase.cnt[2] = tune$phase.cnt[2]+1
      
      # Fit a logistic regression to the response and generate new random probe point
      YY          = matrix(c(tune$m.window[1:tune$W.burnin], tune$n.window[1:tune$W.burnin]), tune$W.burnin, 2)
      tune$b.tune = suppressWarnings(glm(y ~ x, data=data.frame(y=YY[,1]/YY[,2],x=tune$delta.window[1:tune$W.burnin]), family=binomial, weights=YY[,2]))
      
      probe.p     = runif(1)*0.7 + 0.15
      tune$delta  = exp( -(log(1/probe.p-1) + tune$b.tune$coefficients[1])/tune$b.tune$coefficients[2] )
      tune$delta  = min(tune$delta, DELTA.MAX);
      tune$delta  = max(tune$delta, DELTA.MIN);
      
      if (tune$delta > tune$delta.max)
      {
        tune$delta.max = tune$delta
      }
      else if (tune$delta < tune$delta.min)
      {
        tune$delta.min = tune$delta
      }
      
      if (tune$delta == tune$delta.max || tune$delta == tune$delta.min)
      {
        tune$delta = exp(runif(1)*(log(tune$delta.max) - log(tune$delta.min)) + log(tune$delta.min))
      }
    }
    
    #
    tune$M = 0
    tune$D = 0
  }    

  # If we have reached last sample of burn-in, select a suitable delta
  if (tune$iter == tune$burnin)
  {
    # If the algorithm has not explored the space sufficiently, give an error
    if (tune$phase.cnt[2] < 100)
    {
      stop('Metropolis-Hastings sampler has not explored the step-size space sufficiently; please increase the number of burnin samples');
    }
    
    #tune$b.tune = glmfit(tune.delta_window(1:tune.W_burnin), [tune.m_window(1:tune.W_burnin), tune.n_window(1:tune.W_burnin)], 'binomial');
    YY          = matrix(c(tune$m.window[1:tune$W.burnin], tune$n.window[1:tune$W.burnin]), tune$W.burnin, 2)
    tune$b.tune = suppressWarnings(glm(y ~ x, data=data.frame(y=YY[,1]/YY[,2],x=tune$delta.window[1:tune$W.burnin]), family=binomial, weights=YY[,2]))
    
    # Select the final delta to use
    tune$delta  = exp( -(log(1/0.55-1) + tune$b.tune$coefficients[1])/tune$b.tune$coefficients[2] )
    #if (tune.delta == 0 || isinf(tune.delta))
    #    tune.delta = log(tune.delta_max)/2;
    #end
    
    if (tune$display)
    {
      #df = data.frame(log.delta=tune$delta.window[1:tune$W.burnin], p=YY[,1]/YY[,2])
      #df.2 = data.frame(y=predict(tune$b.tune,newdata=data.frame(x=seq(log(tune$delta.min)-5,log(tune$delta.max)+5,length.out=100)),type="response"), x=seq(log(tune$delta.min)-5,log(tune$delta.max)+5,length.out=100))
      #print(ggplot(df, aes(x=log.delta,y=p)) + geom_point() + geom_line(data=df.2,aes(x=x,y=y,color="red")) + labs(title=paste("mGrad Burnin Tuning: Final delta = ", sprintf("%.3g",tune$delta), sep=""), x="log(delta)", y="Estimated Probability of Acceptance") + theme(legend.position = "none"))
    }
  }
  
  # Done
  tune    
}