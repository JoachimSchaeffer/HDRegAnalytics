# Script for Fused Lasso on the Parabola Exmaple
# Using the genlasso function from the genlasso package.
# (Alternatively, fusedlasso vcould be used as well)
# Copyright: Joachim Schaeffer joachim.schaeffer@posteo.de

## CLEAN UP
rm(list = ls()) # Clear packagess
tryCatch(p_unload(all), error=function(e){print("Skip clearing plots, probably no addons!")})  # Unload add-ons
tryCatch(dev.off(), error=function(e){print("Skip clearing plots, probably no plots!")})  # Clear all plots
cat("\014")  # Clear console

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
source(paste(path_base, "src/utils.R", sep = ""))

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
matplot(x_parab, coef(fit, s = 1)[2:(p + 1),],  type = "l")

cvfit_rr <-
  cv.glmnet(X_,
            y_,
            alpha = 0,
            lambda = lambda_seq,
            standardize = F)
plot(cvfit_rr)

plot(
  x_parab,
  coef(cvfit_rr, s = "lambda.1se", excact = T)[2:(p + 1),],
  type = 'l',
  ylab = "",
  xlab = ""
)

y_pred <- predict(cvfit_rr, X_, s = cvfit_rr$lambda.1se)
plot_one_set_predictions(y, y_pred, y_list)

# Fused Lasso (1D Fused Lasso)
x <-
  c(rep(0, p - 2), 1,-1, rep(0, p - 1))
D_step <- toeplitz2(x, p, p)
# D_step_sparse <- as(D_step, "sparseMatrix")
fl <-
  genlasso(y_,
           X_,
           D_step, )
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
nfolds <- 10 # Debugging, increase to 10
N <- nrow(X_)
foldid <- sample(rep(seq(nfolds), length = N))
op <- options(nwarnings = 10000) # Keep all warnings!

genlasso.fit <- genlasso(y = y_,
                         X = X_,
                         D = D_step,)

## Evaluate each lambda on each fold:
fold.lambda.losses <-
  tapply(seq_along(foldid), foldid, function(fold.indices) {
    fold.genlasso.fit <- genlasso(y = y_[-fold.indices],
                                  X = X_[-fold.indices,],
                                  D = D_step)

    fold.genlasso.preds <- predict(fold.genlasso.fit,
                                   lambda = genlasso.fit$lambda,
                                   #$
                                   Xnew = X_[fold.indices, ])$fit 
    lambda.losses <-
      colMeans((fold.genlasso.preds - y_[fold.indices]) ^ 2)
    return (lambda.losses)
  })
# CV loss for each lambda:
cv.lambda.losses <- colMeans(do.call(rbind, fold.lambda.losses))
cv.genlasso.lambda.min <-
  genlasso.fit$lambda[which.min(cv.lambda.losses)]

lambda_min_loss <-
  fl$lambda[which(cv.lambda.losses == min(cv.lambda.losses), arr.ind = TRUE)]

print("Min Lambda CV")
print(lambda_min_loss)

loss_larger_than_1se <-
  cv.lambda.losses > (std(cv.lambda.losses) + min(cv.lambda.losses))
id_1se <- min(which(loss_larger_than_1se == FALSE))
lambda_fl_cv_1se <- fl$lambda[id_1se]

print("Min Lambda 1SE CV")
print(lambda_fl_cv_1se)

# Predict:
lambda <- lambda_fl_cv_1se
y_pred_fl_cv <-
  predict(genlasso.fit, lambda = lambda, Xnew = X_)$fit
coeff_fl_cv = coef(genlasso.fit, lambda = lambda, exact = T)
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
    coef_1se_cv_rr = coef(cvfit_rr, s = "lambda.1se", excact = T)[2:(p + 1),] * y_list$std,
    coef_1se_cv_fused_lasso = unname(coef(genlasso.fit, lambda = lambda_fl_cv_1se, exact = T)$beta * y_list$std)
  )
write.csv(df_reg_coef, paste(path_base, "data/r/parab_n_reg_coeff.csv", sep = ""))
