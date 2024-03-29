# Script for Fused Lasso on the LFP Data with Cycle Life Response
# Copyright: Joachim Schaeffer joachim.schaeffer@posteo.de
# Run time for the entire script including CV 15-20 mintues.

## CLEAN UP
rm(list = ls())
tryCatch(
  p_unload(all),
  error = function(e) {
    print("Skip unloading addons, probably no addons!")
  }
)
tryCatch(
  dev.off(),
  error = function(e) {
    print("Skip clearing plots, probably no plots!")
  }
)
cat("\014")
set.seed(42)

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
# FUSED LASSO
# 1. D1: Sparsity in difference penalty
# 2. D2: "Trendfiltering" k=2
# 3. D4: Standardize + D1

x <-
  c(rep(0, dim(X_train)[2] - 2), 1, -1, rep(0, dim(X_train)[2] - 1))
D1 <- toeplitz2(x, 1000, 1000)
#D1_sparse <- as(D1, "sparseMatrix")
fl <-
  genlasso(y_train_,
           X_train_,
           D1)
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

x <-
  c(rep(0, dim(X_train)[2] - 3), 0.5, -1, 0.5, rep(0, dim(X_train)[2] - 1))
D2 <- toeplitz2(x, 1000, 1000)

sd_train <- apply(X_train_, 2, sd)
X_train_std <- scale(X_train_, center = F, scale = sd_train)
X_test1_std <- scale(X_test1_, center = F, scale = sd_train)
X_test2_std <- scale(X_test2_, center = F, scale = sd_train)
fl_std_p1 <- genlasso(y_train_, X_train_std, D1)
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
# The fused model, can regularize the noise that would otherwise blow up!

# CV for choosing the regularization parameter!
# The following CV code is adapted from locobros answer:
# https://stats.stackexchange.com/questions/198361/why-i-am-that-unsuccessful-with-predicting-with-generalized-lasso-genlasso-gen
# Generate folds, should be the same for all CV runs!
nfolds <- 10
N <- nrow(X_train_)
foldid <- sample(rep(seq(nfolds), length = N))

cv_list = cv_genlasso(X_train_, y_train_, D1, foldid, y_train_list)
coeff_cv = coef(cv_list$genlasso.fit , lambda = cv_list$lambda.min)
plot(x_lfp, coeff_cv$beta, type = "l")
predict_plot_lfp(
  cv_list$genlasso.fit,
  cv_list$lambda.min,
  X_train_,
  X_test1_,
  X_test2_,
  y_train_list,
  model = "genlasso"
)
# An issue here is that the "longest" and "shortest" lived batteries
# can blow up the atandard deviation significantly. --> Using the min cv value.
## SAVE REGRESSION COEFFICIENTS
df_reg_coef <-
  data.frame(
    coef_D1_1se = coef(cv_list$genlasso.fit , lambda = cv_list$lambda.1se)$beta * y_train_list$std,
    coef_D1_cv = coef(cv_list$genlasso.fit , lambda = cv_list$lambda.min)$beta * y_train_list$std
  )
write.csv(
  df_reg_coef,
  paste(
    path_base,
    "regression_in_R/lfp_cl_D1_cv_reg_coeff.csv",
    sep = ""
  )
)

lm_df <- as.data.frame(cv_list$lossmatrix)
names(lm_df) <- cv_list$lambda_vals
write.csv(lm_df,
          paste(
            path_base,
            "regression_in_R/lfp_cl_D1_cv_lossmatrix.csv",
            sep = ""
          ))

# Now for the Std data!
# lambda_seq <- linspace(5, 0.01, 500)
#lambda_seq = lambda_seq,
cv_list_D1_std = cv_genlasso(
  X_train_std,
  y_train_,
  D1,
  foldid,
  y_train_list,
  minlam = c(1e-2, 1e-3),
  maxsteps = c(2000, 2000),

)
coeff_D1_std_cv = coef(cv_list_D1_std$genlasso.fit , lambda = cv_list_D1_std$lambda.min)
plot(x_lfp, coeff_D1_std_cv$beta, type = "l")
predict_plot_lfp(
  cv_list_D1_std$genlasso.fit,
  cv_list_D1_std$lambda.min,
  X_train_std,
  X_test1_std,
  X_test2_std,
  y_train_list,
  model = "genlasso"
)
## SAVE REGRESSION COEFFICIENTS
df_reg_coef <-
  data.frame(
    coef_D1_std_1se = coef(cv_list_D1_std$genlasso.fit , lambda = cv_list_D1_std$lambda.1se)$beta *
      y_train_list$std,
    coef_D1_std_cv = coef(cv_list_D1_std$genlasso.fit , lambda = cv_list_D1_std$lambda.min)$beta *
      y_train_list$std
  )
write.csv(
  df_reg_coef,
  paste(
    path_base,
    "regression_in_R/lfp_cl_D1_std_cv_reg_coeff.csv",
    sep = ""
  )
)

lm_df <- as.data.frame(cv_list_D1_std$lossmatrix)
names(lm_df) <- cv_list_D1_std$lambda_vals
write.csv(
  lm_df,
  paste(
    path_base,
    "regression_in_R/lfp_cl_D1_std_cv_lossmatrix.csv",
    sep = ""
  )
)

# Now for the D2
cv_list_D2 = cv_genlasso(
  X_train_,
  y_train_,
  D2,
  foldid,
  y_train_list,
  minlam = c(1e-4, 1e-5),
  maxsteps = c(5000, 5000)
)
coeff_D2_cv = coef(cv_list_D2$genlasso.fit , lambda = cv_list_D2$lambda.min)
plot(x_lfp, coeff_D2_cv$beta, type = "l")
predict_plot_lfp(
  cv_list_D2$genlasso.fit,
  cv_list_D2$lambda.min,
  X_train_,
  X_test1_,
  X_test2_,
  y_train_list,
  model = "genlasso"
)

## SAVE REGRESSION COEFFICIENTS
df_reg_coef <-
  data.frame(
    coef_D2_1se = coef(cv_list_D2$genlasso.fit , lambda = cv_list_D2$lambda.1se)$beta * y_train_list$std,
    coef_D2_cv = coef(cv_list_D2$genlasso.fit , lambda = cv_list_D2$lambda.min)$beta * y_train_list$std
  )
write.csv(
  df_reg_coef,
  paste(
    path_base,
    "regression_in_R/lfp_cl_D2_cv_reg_coeff.csv",
    sep = ""
  )
)

lm_df <- as.data.frame(cv_list_D2$lossmatrix)
names(lm_df) <- cv_list_D2$lambda_vals
write.csv(lm_df,
          paste(
            path_base,
            "regression_in_R/lfp_cl_D2_cv_lossmatrix.csv",
            sep = ""
          ))
