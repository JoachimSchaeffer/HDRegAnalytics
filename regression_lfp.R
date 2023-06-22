# Script for, testing and debugging.
# 1. Testing different regression methods
# 2. Regression coefficient variance estimation
# Copyright: Joachim Schaeffer joachim.schaeffer@posteo.de
# CLEAN UP
rm(list = ls()) # Clear packages
p_unload(all)  # Unload add-ons
dev.off()  # Clear all plots
cat("\014")  # Clear console
# Clear mind :)
## Packages
pacman::p_load(pacman,
               MASS,
               bayesreg,
               glmnet,
               rio,
               ggplot2,
               pracma,
               genlasso,
               Matrix,
               resample)

path_base <- "~/Documents/PhD/02Research/01Papers/03Nullspace/HDFeat/"
source(paste(path_base, "src/utils.R", sep=""))

## Load Data
path <- paste(path_base, "data/lfp_slim.csv", sep = "")

lfp_data = import(path)
# head(lfp_data)

## Construct Data Matrices
train_id <- lfp_data[, 1004] == 0
test1_id <- lfp_data[, 1004] == 1
test1_id[43] <- F # Removing the shortest lived outlier battery!
test2_id <- lfp_data[, 1004] == 2
y_mean <-
  import(paste(path_base, "data/lfp_y_mean.csv", sep = ""))
y_cm <-
  import(paste(path_base, "data/lfp_y_cm.csv", sep = ""))

X <- unname(as.matrix(rev(lfp_data[, 2:1001])))
X_list <- centerXtrain(X)
X_ = X_list$X_

X_train <- X[train_id,]
X_test1 <- X[test1_id,]
X_test2 <- X[test2_id,]

X_train_list <- centerXtrain(X_train)
X_train_ <- X_train_list$X_
X_test1_ <- centerXtest(X_test1, X_train_list)
X_test2_ <- centerXtest(X_test2, X_train_list)

y <-  lfp_data[, 1002]
y_train <- log(unname(as.matrix(y[train_id])))
y_test1 <- log(unname(as.matrix(y[test1_id])))
y_test2 <- log(unname(as.matrix(y[test2_id])))

y_train_list <- standardize_y_train(y_train)
y_train_ <- y_train_list$y_std

y_train_mean = rowMeans(X_train)
y_train_mean_ <- y_train_mean - mean(y_train_mean)

x_lfp <-  seq(2, 3.5, length.out = 1000)

## Data Visualization
matplot(
  t(X),
  type = "l",
  ylab = "DeltaQ (V)",
  xlab = "Voltage (V)",
  main = "LFP Data Set"
)
matplot(
  t(X_train_),
  type = "l",
  ylab = "DeltaQ (V)",
  xlab = "Voltage (V)",
  main = "LFP Data Set"
)

# REGRESSION SECTION
# Run regression with glmnet (alpha = 0 is ridge regression!)
lambda_seq <- logseq(10 ^ -6, 10 ^ 2, n = 2000)
fit <- glmnet(
  X_train_,
  y_train_,
  lambda = lambda_seq,
  thresh = 1e-10,
  alpha = 0,
  standardize = F,
  intercept = F,
  excact = T,
)
# plot(fit)
# print(fit)
# coef(fit)
# matplot(x_lfp, coef(fit, s = 1)[2:1001, ],  type = "l")
# matplot(
#  x_lfp,
#  predict(
#    x = X_train_,
#    y = y_train_,
#    fit,
#    type = "coefficient",
#    s = 0.1,
#    exact = T
#  )[2:1001, ],
#  type = "l"
# )

cvfit <-
  cv.glmnet(
    X_train_,
    y_train_,
    alpha = 0,
    lambda = lambda_seq,
    standardize = F
  )
plot(cvfit)
cvfit$lambda.min
cvfit$lambda.1se
plot(
  x_lfp,
  coef(cvfit, s = "lambda.1se", excact = T)[2:1001, ],
  type = 'l',
  ylab = "",
  xlab = ""
)

# Calculate the RR coefficients manually
# w1 <- solve(t(X_train_)%*%X_train_+41*diag(1000),t(X_train_)%*%(y_train_))[,1]
# w2 <- as.vector(coef(fit, x= X_train_, y=y_train_, s = 1, exact = TRUE))[-1]
# cbind(w1[1:10], w2[1:10])
# --> Regression coefficients are the same. Yuhuu. Lets move on.
# More details about differences in the objective function can be found here:
# https://stats.stackexchange.com/questions/129179/why-is-glmnet-ridge-regression-giving-me-a-different-answer-than-manual-calculat

# Calculate stats on tests sets
y_pred <- predict(cvfit, X_train_, s = cvfit$lambda.1se)
y_pred_test1 <- predict(cvfit, X_test1_, s = cvfit$lambda.1se)
y_pred_test2 <- predict(cvfit, X_test2_, s = cvfit$lambda.1se)
plot_predictions(y_train,
                 y_pred,
                 y_test1,
                 y_pred_test1,
                 y_test2,
                 y_pred_test2,
                 y_train_list)

# Try fused lasso (1D Fused Lasso)
x <-
  c(rep(0, dim(X_train)[2] - 2), 1, -1, rep(0, dim(X_train)[2] - 1))
D_step <- toeplitz2(x, 1000, 1000)
D_step_sparse <- as(D_step, "sparseMatrix")

# Its an interplay of the scale of X and the epsilon that will be added as ridge penalty.
# Actually this is rather a fused EN. Even though the authors did not make it explicit,
# by setting eps you can actually run a fused version of the elastic net.
# eps = 1e-4, work well for the data in the original scale. If the data is rescaled,
# might be worth to reconsider how eps.
# --> Killer for functional data.

fl <-
  fusedlasso(
    y_train_,
    X_train_,
    D_step_sparse,
    gamma = 100,
    minlam = 1e-7,
    eps = 1e-4,
  )
lambda_val <- 0.000002
plot(fl)
coeff_fused_lasso = coef(fl, lambda = lambda_val, exact = T)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
# Check the predictions. (put the prediction stuff in a function for easy calling!)
# Interesting coefficients!
y_pred <- predict(fl, lambda = lambda_val, Xnew = X_train_)$fit
y_pred_test1 <-
  predict(fl, lambda = lambda_val, Xnew = X_test1_)$fit
y_pred_test2 <-
  predict(fl, lambda = lambda_val, Xnew = X_test2_)$fit
plot_predictions(y_train,
                 y_pred,
                 y_test1,
                 y_pred_test1,
                 y_test2,
                 y_pred_test2,
                 y_train_list)
# gamma = 0, minlam = 1e-7, eps = 1e-4, %% lambda = 0.005 
# gamma = 100, minlam = 1e-7, eps = 1e-4,) lambda_val <- 0.000002
# TBH: Pretty neat results. It struggles for the long lived cells, but that's well known.

x <-
  c(rep(0, dim(X_train)[2] - 3), 0.5, -1, 0.5, rep(0, dim(X_train)[2] - 1))
D_p2 <- toeplitz2(x, 1000, 1000)
D_p2_sparse <- as(D_p2, "sparseMatrix")
# Its important to make sure that the the rowsum is 1 // that the toeplitz matirx
# is rescaled. Need to build some more intuition aroud how this actually works.
fl_p2 <-
  fusedlasso(y_train_,
             X_train_,
             D_p2_sparse,
             gamma = 0.1,
             minlam = 1e-7,)

lambda_val <- 0.0001
coeff_fused_lasso = coef(fl_p2, lambda = lambda_val)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
y_pred <- predict(fl_p2, lambda = lambda_val, Xnew = X_train_)$fit
y_pred_test1 <-
  predict(fl_p2, lambda = lambda_val, Xnew = X_test1_)$fit
y_pred_test2 <-
  predict(fl_p2, lambda = lambda_val, Xnew = X_test2_)$fit
plot_predictions(y_train,
                 y_pred,
                 y_test1,
                 y_pred_test1,
                 y_test2,
                 y_pred_test2,
                 y_train_list)


# Rerun with standardization.\
# Large gamma will converge toward classical lasso as the lasso penalizatoin
# is more important.
# gamma = lasso_lambda/gamma_lambda
# When standardized, it's can be benefifical for this data to move away from the lasso
# increase eps to push it towards en/rr.
X_train_std = as.matrix(scale(data.frame(X_train_)))
fl_std_step <-
  fusedlasso(
    y_train_,
    X_train_std,
    D_step_sparse,
    gamma = 1,
    minlam = 1e-4,
    eps = 1,
    rtol = 1e-11,
  )
plot(fl_std_step)
# matplot(t(X_train_std), type='l')
coeff_fused_lasso = coef(fl_std_step, lambda = 0.1)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
# --> Standardization blowing up the noise.
# Can still be helpful. but smart rescaling is better.
# Something weird in here, but ignore for now.


# CV 
# CV for chosing the EN parameter might not be trivial because lambdas will be different.
# BuT we could save all the lambdas in a matrix and then choose in the ned based on cv.
# Take care, this function fits a crazy amount of regression fits and can take a few hours depending on your system. 
# Lambda sequence could be optimized. 
# Technically, this is an abuse of the ``genlasso'' function, forcing it to become a "fused elastic net".
# The following CV code is adapted from locobros answer:
# https://stats.stackexchange.com/questions/198361/why-i-am-that-unsuccessful-with-predicting-with-generalized-lasso-genlasso-gen

nfolds <- 10 # Debugging, increase to 10
eps_seq <- logseq(10^-5, 1, n = 30)
n_eps <- length(eps_seq)
eps_ <- mean(eps_seq)
N <- nrow(X_train_)
foldid <- sample(rep(seq(nfolds), length = N))
# First run to determine lambda sequence automatically 
fusedlasso.fit <- fusedlasso(
  y_train_,
  X_train_,
  D_step_sparse,
  gamma = 0,
  minlam = 1e-5,
  eps = eps_,
  rtol = 1e-11,
)
op <- options(nwarnings = 10000) # Keep all warnings! 
fold.lambda.losses <- vector("list", n_eps)
for(i in 1:n_eps) {
  fold.lambda.losses[[i]] <-
    tapply(seq_along(foldid), foldid, function(fold.indices) {
      fold.fusedlasso.fit <- fusedlasso(
        y_train_[-fold.indices],
        X_train_[-fold.indices, ],
        D_step_sparse,
        gamma = 0,
        minlam = 1e-5,
        eps = eps_seq[i],
        rtol = 1e-11,
      )
      ## length(fold.indices)-by-length(cv.genlasso.fit$lambda) matrix, with
      ## predictions for this fold:
      ## $
      fold.fusedlasso.preds <- predict(fold.fusedlasso.fit,
                                       lambda = fl$lambda,
                                       #$
                                       Xnew = X_train_[fold.indices, ])$fit #$
      lambda.losses <-
        sqrt(colMeans((fold.fusedlasso.preds - y_train_[fold.indices]) ^ 2))
      return (lambda.losses)
    })
}
# Loop through the results, put them in new list. Column names: alpha, rows: eps
cv.lambda.losses_mean <- matrix(,nrow = length(eps_seq), ncol = length(fl$lambda))
cv.lambda.losses_sd <- matrix(,nrow = length(eps_seq), ncol = length(fl$lambda))

for(i in 1:n_eps) {
  disp(i)
  cv.lambda.losses_mean[i, ] <- colMeans(do.call(rbind, fold.lambda.losses[[i]]))
  cv.lambda.losses_sd[i, ] <- colStdevs(do.call(rbind, fold.lambda.losses[[i]]))
}

# matplot(cv.lambda.losses_var, type = "l")
# matplot(cv.lambda.losses_mean, type = "l")

# Pick the best!
lampba_min_loss <-  fl$lambda[which(cv.lambda.losses_mean == min(cv.lambda.losses_mean), arr.ind = TRUE)[2]]
eps_min_loss <- eps_seq[which(cv.lambda.losses_mean == min(cv.lambda.losses_mean), arr.ind = TRUE)[1]]
fl <- fusedlasso(
  y_train_,
  X_train_,
  D_step_sparse,
  gamma = 0,
  minlam = 1e-5,
  eps = eps_min_loss,
  rtol = 1e-11,
)
coeff_fused_lasso = coef(fl , lambda = lampba_min_loss)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
## Predict:
y_pred <- predict(fl, lambda = lampba_min_loss, Xnew = X_train_)$fit
y_pred_test1 <-
  predict(fl, lambda = lampba_min_loss, Xnew = X_test1_)$fit
y_pred_test2 <-
  predict(fl, lambda = lampba_min_loss, Xnew = X_test2_)$fit
plot_predictions(y_train,
                 y_pred,
                 y_test1,
                 y_pred_test1,
                 y_test2,
                 y_pred_test2,
                 y_train_list)

## Test the SNR rescaled data (we don't expect a large benefit here!)
# Lambda 10 + eps 1e-4: Awesome!
# 
lfp_snr_smooth_dB <- import(paste(path_base, "HDFeat/data/lfp_snr_smooth_dB.csv", sep = ""))
mean_sd_list <- mean_sd_fused_lasso(X_train)
sd <- mean_sd_list$sd
lfp_snr_smooth_dB <- (lfp_snr_smooth_dB-min(lfp_snr_smooth_dB)+1e-4)/(max(lfp_snr_smooth_dB)-min(lfp_snr_smooth_dB));
scale_factor_snr_db <- as.matrix(sd/(lfp_snr_smooth_dB))

X_train_snr_ <- scale(X_train, mean_sd_list$mean, scale_factor_snr_db)
X_test1_snr_ <- scale(X_test1, mean_sd_list$mean, scale_factor_snr_db)
X_test2_snr_ <- scale(X_test2, mean_sd_list$mean, scale_factor_snr_db)

fl <-
  fusedlasso(
    y_train_,
    X_train_snr_,
    D_step_sparse,
    gamma = 0,
    minlam = 1e-7,
    eps = 0.1,
    rtol = 1e-11,
    # btol = 1e-11
  )
#, eps = 0.1)#, minlam = 0.000001)
plot(fl)
coeff_fused_lasso = coef(fl, lambda = 1, exact = T)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
# Check the predictions. (put the prediction stuff in a function for easy calling!)
# Interesting coefficients!
lambda_val <- 1
y_pred <- predict(fl, lambda = lambda_val, Xnew = X_train_snr_)$fit
y_pred_test1 <-
  predict(fl, lambda = lambda_val, Xnew = X_test1_snr_)$fit
y_pred_test2 <-
  predict(fl, lambda = lambda_val, Xnew = X_test2_snr_)$fit
plot_predictions(y_train,
                 y_pred,
                 y_test1,
                 y_pred_test1,
                 y_test2,
                 y_pred_test2,
                 y_train_list)



# CLEAN UP
rm(list = ls()) # Clear packages
p_unload(all)  # Unload add-ons
dev.off()  # Clear all plots
cat("\014")  # Clear console
# Clear mind :)
