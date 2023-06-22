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
y <-  unname(as.matrix(parab_data[, (p + 1)]))
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
matplot(x_parab, coef(fit, s = 1)[2:(p + 1),],  type = "l")

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
  coef(cvfit, s = "lambda.1se", excact = T)[2:(p + 1),],
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
# D_step_sparse <- as(D_step, "sparseMatrix")
fl <-
  genlasso(y_,
           X_,
           D_step, )
plot(fl)
lambda_val <- 0.05
coeff_fused_lasso = coef(fl, lambda = lambda_val, exact = T)
plot(x_parab, coeff_fused_lasso$beta * y_list$std, type = "l")
# Check the predictions. (put the prediction stuff in a function for easy calling!)
# Interesting coefficients!
y_pred <- predict(fl, lambda = lambda_val, Xnew = X_)$fit
# Scatter the predictions
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
    ## length(fold.indices)-by-length(cv.genlasso.fit$lambda) matrix, with
    ## predictions for this fold:
    ## $
    fold.genlasso.preds <- predict(fold.genlasso.fit,
                                   lambda = genlasso.fit$lambda,
                                   #$
                                   Xnew = X_[fold.indices, ])$fit #$
    lambda.losses <-
      colMeans((fold.genlasso.preds - y_[fold.indices]) ^ 2)
    return (lambda.losses)
  })
## CV loss for each lambda:
cv.lambda.losses <- colMeans(do.call(rbind, fold.lambda.losses))
cv.genlasso.lambda.min <-
  genlasso.fit$lambda[which.min(cv.lambda.losses)]


## Predict:
cv.genlasso.lambda.min.pred <- predict(genlasso.fit,
                                       lambda = cv.genlasso.lambda.min,
                                       Xnew = cbind(1,test.x))$fit #$

# Pick the best!
lambda_min_loss <-
  fl$lambda[which(cv.lambda.losses == min(cv.lambda.losses), arr.ind = TRUE)]
# Print! 
print("Min Loss CV")
print(lambda_min_loss)

# Pick the best with 1se
loss_larger_than_1se <- cv.lambda.losses > (std(cv.lambda.losses) + min(cv.lambda.losses))
# In this idealized examaple, all but one are within the 1se.
# --> Basically all somewhat reasonable lambdas will do great.
id_1se <- min(which(loss_larger_than_1se == FALSE))
lambda_dl_cv_1se <- fl$lambda[id_1se]

## Predict:
lambda <- lambda_dl_cv_1se
y_pred_fl_cv <- predict(genlasso.fit, lambda = lambda, Xnew = X_)$fit
coeff_fl_cv = coef(genlasso.fit, lambda = lambda, exact = T)
# Axis limtis necessary, otherwise R will show numeric noise! 
plot(x_parab, coeff_fl_cv$beta * y_list$std, type = "l", ylim = c(0.001, 0.008))

plot_one_set_predictions(y, y_pred_fl_cv, y_list)


# ToDo: Check with fusedlasso implementeation. 
# Save the regression coefficients. 
# Put them in. amtarix, next to each other.
# plot the RR predictions.