# Script for Fused Lasso on the LFP Data with Cycle Life Response
# Copyright: Joachim Schaeffer joachim.schaeffer@posteo.de

## CLEAN UP
rm(list = ls())
tryCatch(
  p_unload(all),
  error = function(e) {
    print("Skip clearing plots, probably no addons!")
  }
)
tryCatch(
  dev.off(),
  error = function(e) {
    print("Skip clearing plots, probably no plots!")
  }
)
cat("\014")

## LOADING
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

path_base <-
  "~/Documents/PhD/02Research/01Papers/03Nullspace/HDFeat/"
source(paste(path_base, "regression_in_R/utils.R", sep = ""))

## Load Data
path <- paste(path_base, "data/lfp_slim.csv", sep = "")

lfp_data = import(path)
# head(lfp_data)

## Construct Data Matrices
train_id <- lfp_data[, 1004] == 0
test1_id <- lfp_data[, 1004] == 1
test1_id[43] <- F # Removing the shortest lived outlier battery!
test2_id <- lfp_data[, 1004] == 2

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
lambda_seq <- logseq(10 ^ -8, 0.1, n = 1000)
cvfit <-
  cv.glmnet(
    X_train_,
    y_train_,
    n_folds = 10,
    alpha = 0,
    lambda = lambda_seq,
    standardize = F,
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

# Calculate the RR coefficients manually to check how it's done in glmnet,
# n = dim(y_train_)[1]
# y_std = sqrt(var(y_train_)*(n-1)/n)[1,1]
# lambda = 2
# w1 <- solve(t(X_train_)%*%X_train_+lambda*diag(1000),t(X_train_)%*%(y_train_))[,1]
# fit_glmnet_test <- glmnet(X_train_, y_train_, alpha=0, standardize = F, intercept = FALSE, thresh = 1e-20)
# w2 <- as.vector(coef(fit_glmnet_test, x= X_train_, y=y_train_, s = lambda*y_std/41, exact = TRUE))[-1]
# cbind(w1[1:10], w2[1:10])
# --> Regression coefficients are the same. Yuhuu. Lets move on.
# More details about differences in the objective function can be found here:
# https://stats.stackexchange.com/questions/129179/why-is-glmnet-ridge-regression-giving-me-a-different-answer-than-manual-calculat

# Calculate stats on tests sets
disp("Evaluating RR MIN CV ...")
predict_plot_lfp(
  cvfit,
  lambda_val = cvfit$lambda.min,
  X_train_,
  X_test1_,
  X_test2_,
  y_train_list,
  model = "cvfit"
)
disp("Evaluating RR 1SE CV ...")
predict_plot_lfp(
  cvfit,
  lambda_val = cvfit$lambda.1se,
  X_train_,
  X_test1_,
  X_test2_,
  y_train_list,
  model = "cvfit"
)

# Run CV on the elastic net:
# Code Snippets from: https://rpubs.com/jmkelly91/881590
models <- list()
results <- data.frame()
for (i in 0:20) {
  name <- paste0("alpha", i / 20)
  disp(name)
  models[[name]] <-
    cv.glmnet(
      X_train_,
      y_train_,
      n_folds = 10,
      alpha = i / 20,
      lambda = lambda_seq,
      standardize = F,
    )
  # min_cvm_id <- which(min(models[[name]]$cvm) == models[[name]]$cvm)
  mse <- min(models[[name]]$cvm)
  lambda_min <- models[[name]]$lambda.min
  lambda_1se <- models[[name]]$lambda.1se
  ## Store the results
  temp <-
    data.frame(
      alpha = i / 20,
      min_mse = mse,
      lambda_min = lambda_min,
      lambda_1se = lambda_1se,
      name = name
    )
  results <- rbind(results, temp)
}
best_id <- which(min(results$min_mse) == results$min_mse)
results_best <- results[best_id, ]
plot(
  x_lfp,
  coef(
    models[[best_id]],
    lambda = results_best$lambda_min,
    alpha = results_best$alpha,
    excact = T
  )[2:1001, ],
  type = 'l',
  ylab = "",
  xlab = ""
)
predict_plot_lfp(
  models[[best_id]],
  lambda_val = results_best$lambda_1se,
  X_train_,
  X_test1_,
  X_test2_,
  y_train_list,
  model = "cvfit"
)
predict_plot_lfp(
  models[[best_id]],
  lambda_val = results_best$lambda_min,
  X_train_,
  X_test1_,
  X_test2_,
  y_train_list,
  model = "cvfit"
)


# FUSED LASSO
# 1. D1: Sparsity in difference penalty
# 2. D2: "Trendfiltering" k=2
# 3. D3: "Trendfiltering" k=2
# 4. D4: Standardize + D1
x <-
  c(rep(0, dim(X_train)[2] - 2), 1, -1, rep(0, dim(X_train)[2] - 1))
D_step <- toeplitz2(x, 1000, 1000)
#D_step_sparse <- as(D_step, "sparseMatrix")
fl <-
  genlasso(y_train_,
           X_train_,
           D_step)
lambda_val <- 0.001
coeff_fused_lasso = coef(fl, lambda = lambda_val, exact = T)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
predict_plot_lfp(fl,
                 lambda_val,
                 X_train_,
                 X_test1_,
                 X_test2_,
                 y_train_list,
                 model = "genlasso")
# Pretty neat results. It struggles for the long lived cells, but that's well known.
# Other experimental values that can be tried with the fusedlasso function.
# However, even though the genlasso function is slower, it is more stable for this data set.
# gamma = 0, minlam = 1e-7, eps = 1e-4, %% lambda = 0.005
# gamma = 100, minlam = 1e-7, eps = 1e-4,) lambda_val <- 0.000002

x <-
  c(rep(0, dim(X_train)[2] - 3), 0.5, -1, 0.5, rep(0, dim(X_train)[2] - 1))
D_p2 <- toeplitz2(x, 1000, 1000)
fl_p2 <- genlasso(y_train_, X_train_, D_p2)
lambda_val <- 1.1
coeff_fused_lasso = coef(fl_p2, lambda = lambda_val)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
predict_plot_lfp(fl_p2,
                 lambda_val,
                 X_train_,
                 X_test1_,
                 X_test2_,
                 y_train_list,
                 model = "genlasso")

x <-
  c(rep(0, dim(X_train)[2] - 4), 0.25,-0.75, 0.75,-0.25, rep(0, dim(X_train)[2] - 1))
D_p3 <- toeplitz2(x, 1000, 1000)
fl_p3 <- genlasso(y_train_, X_train_, D_p3)
lambda_val <- 1.1
coeff_fused_lasso = coef(fl_p3, lambda = lambda_val)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
predict_plot_lfp(fl_p3,
                 lambda_val,
                 X_train_,
                 X_test1_,
                 X_test2_,
                 y_train_list,
                 model = "genlasso")
# BATMAN!

sd_train <- apply(X_train_, 2, sd)
X_train_std <- scale(X_train_, center=F, scale=sd_train)
X_test1_std <- scale(X_test1_, center=F, scale=sd_train)
X_test2_std <- scale(X_test2_, center=F, scale=sd_train)
fl_std_p1 <- genlasso(y_train_, X_train_std, D_step)
lambda_val = 1
coeff_fused_lasso = coef(fl_std_p1, lambda = lambda_val)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
predict_plot_lfp(fl_std_p1,
                 lambda_val,
                 X_train_std,
                 X_test1_std,
                 X_test2_std,
                 y_train_list,
                 model = "genlasso")
# --> Standardization not terrible of regularization is chosen appropriately!
# The fused model, can regularize the noise that would otherwise appear!

# CV for choosing the regularization parameter!
# The following CV code is adapted from locobros answer:
# https://stats.stackexchange.com/questions/198361/why-i-am-that-unsuccessful-with-predicting-with-generalized-lasso-genlasso-gen

nfolds <- 10
eps_seq <- logseq(10 ^ -5, 1, n = 30)
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
for (i in 1:n_eps) {
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
cv.lambda.losses_mean <-
  matrix(, nrow = length(eps_seq), ncol = length(fl$lambda))
cv.lambda.losses_sd <-
  matrix(, nrow = length(eps_seq), ncol = length(fl$lambda))

for (i in 1:n_eps) {
  disp(i)
  cv.lambda.losses_mean[i, ] <-
    colMeans(do.call(rbind, fold.lambda.losses[[i]]))
  cv.lambda.losses_sd[i, ] <-
    colStdevs(do.call(rbind, fold.lambda.losses[[i]]))
}

# matplot(cv.lambda.losses_var, type = "l")
# matplot(cv.lambda.losses_mean, type = "l")

# Pick the best!
lampba_min_loss <-
  fl$lambda[which(cv.lambda.losses_mean == min(cv.lambda.losses_mean), arr.ind = TRUE)[2]]
eps_min_loss <-
  eps_seq[which(cv.lambda.losses_mean == min(cv.lambda.losses_mean), arr.ind = TRUE)[1]]
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


# Save regression coefficient matrices in the end.





## Maybe remove the following code!

## Test the SNR rescaled data (we don't expect a large benefit here!)
# Lambda 10 + eps 1e-4: Awesome!
#
lfp_snr_smooth_dB <-
  import(paste(path_base, "HDFeat/data/lfp_snr_smooth_dB.csv", sep = ""))
mean_sd_list <- mean_sd_fused_lasso(X_train)
sd <- mean_sd_list$sd
lfp_snr_smooth_dB <-
  (lfp_snr_smooth_dB - min(lfp_snr_smooth_dB) + 1e-4) / (max(lfp_snr_smooth_dB) -
                                                           min(lfp_snr_smooth_dB))

scale_factor_snr_db <- as.matrix(sd / (lfp_snr_smooth_dB))

X_train_snr_ <-
  scale(X_train, mean_sd_list$mean, scale_factor_snr_db)
X_test1_snr_ <-
  scale(X_test1, mean_sd_list$mean, scale_factor_snr_db)
X_test2_snr_ <-
  scale(X_test2, mean_sd_list$mean, scale_factor_snr_db)

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
