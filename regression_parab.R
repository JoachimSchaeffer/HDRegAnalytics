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

path_base <-
  "~/Documents/PhD/02Research/01Papers/03Nullspace/HDFeat/"
source(paste(path_base, "src/utils.R", sep = ""))

## Load Data
path <- paste(path_base, "data/poly_hd_data_n.csv", sep = "")
parab_data = import(path, skip = 16)
## Construct Data Matrices
p = 201
X <- unname(as.matrix(parab_data[, 1:p]))
X_list <- centerXtrain(X)
X_ = X_list$X_
y <-  unname(parab_data[, (p+1)])
y_list <- standardize_y_train(y)
y_ <- y_list$y_std
x_parab <-  seq(1, 3, length.out = p)

## Data Visualization
matplot(
  x_parab,
  t(X_),
  type = "l",
  ylab = "z",
  xlab = "x",
  main = "Prabolas with Noise"
)

# REGRESSION SECTION
# Run regression with glmnet (alpha = 0 is ridge regression!)
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
matplot(x_parab, coef(fit, s = 1)[2:(p+1),],  type = "l")

cvfit <-
  cv.glmnet(X_,
            y_,
            alpha = 0,
            lambda = lambda_seq,
            standardize = F)
plot(cvfit)
cvfit$lambda.min
cvfit$lambda.1se
plot(
  x_parab,
  coef(cvfit, s = "lambda.1se", excact = T)[2:(p+1),],
  type = 'l',
  ylab = "",
  xlab = ""
)

# Calculate stats on tests sets
# y_pred <- predict(cvfit, X_, s = cvfit$lambda.1se)
# plot_predictions(y_, y_pred, y_list)

# Try fused lasso (1D Fused Lasso)
# First we have to define the Matrix D
x <-
  c(rep(0, p - 2), 1,-1, rep(0, p - 1))
D_step <- toeplitz2(x, p, p)
D_step_sparse <- as(D_step, "sparseMatrix")
rm(fl)
fl <-
  fusedlasso(
    y_,
    X_,
    D_step_sparse,
    eps = 0.001,
    gamma = 0.1,
    minlam = 1e-5,
    rtol = 1e-14,
  )
# btol = 1e-11
#, eps = 0.1)#, minlam = 0.000001)
plot(fl)
coeff_fused_lasso = coef(fl, lambda = 0.01, exact = T)
plot(x_parab, coeff_fused_lasso$beta, type = "l")
# Check the predictions. (put the prediction stuff in a function for easy calling!)
# Interesting coefficients!
lambda_val <- 0.005
y_pred <- predict(fl, lambda = lambda_val, Xnew = X_)$fit

# CV
# CV for chosing the EN parameter might not be trivial because lambdas will be different.
# BuT we could save all the lambdas in a matrix and then choose in the ned based on cv.
# Take care, this function fits a crazy amount of regression fits and can take a few hours depending on your system.
# Lambda sequence could be optimized.
# Technically, this is an abuse of the ``genlasso'' function, forcing it to become a "fused elastic net".
# The following CV code is adapted from locobros answer:
# https://stats.stackexchange.com/questions/198361/why-i-am-that-unsuccessful-with-predicting-with-generalized-lasso-genlasso-gen

nfolds <- 10 # Debugging, increase to 10
eps_seq <- logseq(10 ^ -3, 1, n = 30)
n_eps <- length(eps_seq)
eps_ <- mean(eps_seq)
N <- nrow(X_)
foldid <- sample(rep(seq(nfolds), length = N))
# First run to determine lambda sequence automatically
fusedlasso.fit <- fusedlasso(
  y_,
  X_,
  D_step_sparse,
  gamma = 0,
  minlam = 1e-7,
  eps = eps_,
  rtol = 1e-11,
)
op <- options(nwarnings = 10000) # Keep all warnings!
fold.lambda.losses <- vector("list", n_eps)
for (i in 1:n_eps) {
  fold.lambda.losses[[i]] <-
    tapply(seq_along(foldid), foldid, function(fold.indices) {
      fold.fusedlasso.fit <- fusedlasso(
        y_[-fold.indices],
        X_[-fold.indices,],
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
                                       Xnew = X_[fold.indices,])$fit #$
      lambda.losses <-
        sqrt(colMeans((fold.fusedlasso.preds - y_[fold.indices]) ^ 2))
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
  cv.lambda.losses_mean[i,] <-
    colMeans(do.call(rbind, fold.lambda.losses[[i]]))
  cv.lambda.losses_sd[i,] <-
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
  y_,
  X_,
  D_step_sparse,
  gamma = 0,
  minlam = 1e-5,
  eps = eps_min_loss,
  rtol = 1e-11,
)
coeff_fused_lasso = coef(fl , lambda = lampba_min_loss)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")
## Predict:
y_pred <- predict(fl, lambda = lampba_min_loss, Xnew = X_)$fit
