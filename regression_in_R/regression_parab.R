# Script for Fused Lasso on the Parabolic Example
# Using the genlasso function from the genlasso package.
# (Alternatively, the fusedlasso function could be used as well)
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

# Load Data
path <- paste(path_base, "data/poly_hd_data_n.csv", sep = "")
parab_data = import(path, skip = 16)
## Construct Data Matrices
p = 201
X <- unname(as.matrix(parab_data[, 1:p]))
X_list <- centerXtrain(X)
X_ = X_list$X_
y <-  unname(as.matrix(parab_data[, (p + 1)]))
y_list <- standardize_y_train(y)
y_ <- y_list$y_std
x_parab <-  seq(1, 3, length.out = p)

## DATA VISUALIZATION
matplot(
  x_parab,
  t(X_),
  type = "l",
  ylab = "z",
  xlab = "d",
  main = "Parabolas with Noise"
)

# REGRESSION SECTION
# Ridge Regression
lambda_seq <- logseq(10 ^ -6, 10 ^ 2, n = 2000)
fit <- glmnet(
  X_,
  y_,
  lambda = lambda_seq,
  thresh = 1e-10,
  alpha = 0,
  standardize = F,
  intercept = F,
  excact = T,
)
plot(fit)
matplot(x_parab, coef(fit, s = 1)[2:(p + 1), ],  type = "l")

cvfit_rr <-
  cv.glmnet(X_,
            y_,
            alpha = 0,
            lambda = lambda_seq,
            standardize = F)
plot(cvfit_rr)

plot(
  x_parab,
  coef(cvfit_rr, s = "lambda.1se", excact = T)[2:(p + 1), ],
  type = 'l',
  ylab = "",
  xlab = ""
)

y_pred <- predict(cvfit_rr, X_, s = cvfit_rr$lambda.1se)
plot_one_set_predictions(y, y_pred, y_list)

# Fused Lasso (1D Fused Lasso)
x <-
  c(rep(0, p - 2), 1, -1, rep(0, p - 1))
D_step <- toeplitz2(x, p, p)
# D_step_sparse <- as(D_step, "sparseMatrix")
fl <-
  genlasso(y_,
           X_,
           D_step,)
plot(fl)
lambda_val <- 0.05
coeff_fused_lasso = coef(fl, lambda = lambda_val, exact = T)
plot(
  x_parab,
  coeff_fused_lasso$beta * y_list$std,
  type = "l",
  ylim = c(0.001, 0.008)
)
y_pred <- predict(fl, lambda = lambda_val, Xnew = X_)$fit
plot_one_set_predictions(y, y_pred, y_list)
# GREAT! The fused lasso recovers the true coefficients almost perfectly.

# CV only the folds for the lambda
# The following CV code is adapted from locobros answer:
# https://stats.stackexchange.com/questions/198361/why-i-am-that-unsuccessful-with-predicting-with-generalized-lasso-genlasso-gen
cv_list_D1 = cv_genlasso(
  X_,
  y_,
  D_step,
  nfolds = 10,
)

# Predict:
lambda <- cv_list_D1$lambda.1se
y_pred_fl_cv <-
  predict(cv_list_D1$genlasso.fit, lambda = lambda, Xnew = X_)$fit
coeff_fl_cv = coef(cv_list_D1$genlasso.fit, lambda = lambda, exact = T)
# Axis limits necessary, otherwise R will show numeric noise!
plot(
  x_parab,
  coeff_fl_cv$beta * y_list$std,
  type = "l",
  ylim = c(0.001, 0.008)
)
plot_one_set_predictions(y, y_pred_fl_cv, y_list)

## SAVE REGRESSION COEFFICIENTS
df_reg_coef <-
  data.frame(
    coef_1se_cv_rr = coef(cvfit_rr, s = "lambda.1se", excact = T)[2:(p + 1), ] * y_list$std,
    coef_1se_cv_fused_lasso = unname(
      coef(cv_list_D1$genlasso.fit, lambda = cv_list_D1$lambda.1se, exact = T)$beta * y_list$std
    )
  )
write.csv(df_reg_coef,
          paste(path_base, "regression_in_R/parab_n_reg_coeff.csv", sep = ""))
