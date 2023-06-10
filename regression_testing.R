# Script for, testing and debugging.
# 1. Testing different regression methods
# 2. Regression coefficient variacne estimation
# 3. Hypothesis testing
# Copyright: Joachim Schaeffer joachim.schaeffer@posteo.de

## Packages
pacman::p_load(pacman,
               MASS,
               bayesreg,
               glmnet,
               qut,
               rio,
               ggplot2,
               pracma,
               genlasso,
               # "canonical" smoothed lasso
               Matrix)

## Functions (Probably this implementation is somewhat naive as I'm not an R native)
center_X_train <- function(X) {
  mean_ = colMeans(X)
  X_ <- sweep(X, 2, mean_, "-")
  return <- list("X_" = X_, "mean" = mean_)
}

center_X_test <- function(X, X_train_list) {
  X_ <- sweep(X, 2, X_train_list$mean, "-")
  return <-  X_
}

standardize_y_train <- function(y) {
  N = dim(y_train)[1]
  mean_y <- mean(y)
  std_y <- sqrt(var(y) * (N - 1) / N)[1, 1]
  y_std_ <- (y - mean_y) / std_y
  return <-
    list(
      "y_std" = y_std_,
      "mean" = mean_y,
      "std" = std_y,
      "dim" = N
    )
}

standardize_y_test <- function(y, y_train_list) {
  mean_y_train <- y_train_list$mean
  std_y_train <- y_train_list$std
  return <- (y - mean_y_train) / std_y_train
}

rescale_y <- function(y, y_train_list) {
  mean <- y_train_list$mean
  sd_ <- y_train_list$std
  return <-  (y * sd_) + mean
}

plot_predictions <-
  function(y_train,
           y_pred,
           y_test1,
           y_pred_test1,
           y_test2,
           y_pred_test2,
           y_train_list) {
    # Well I guess this is best to do in ggplot!
    train_df <-
      setNames(data.frame(exp(y_train), exp(rescale_y(
        y_pred, y_train_list
      ))),
      c("y_train", "y_train_pred"))
    err_train = rmserr(train_df$y_train, train_df$y_train_pred)
    print(err_train$rmse)
    
    test1_df <-
      setNames(data.frame(exp(y_test1), exp(rescale_y(
        y_pred_test1, y_train_list
      ))), c("y_test1", "y_test1_pred"))
    err_test1 = rmserr(test1_df$y_test1, test1_df$y_test1_pred)
    print(err_test1$rmse)
    
    test2_df <-
      setNames(data.frame(exp(y_test2), exp(rescale_y(
        y_pred_test2, y_train_list
      ))), c("y_test2", "y_test2_pred"))
    err_test2 = rmserr(test2_df$y_test2, test2_df$y_test2_pred)
    print(err_test2$rmse)
    
    
    p <- ggplot() +
      geom_point(data = train_df,
                 aes(x = y_train, y = y_train_pred),
                 color = "#cc0000") +
      geom_point(data = test1_df,
                 aes(x = y_test1, y = y_test1_pred),
                 col = "#00008B") +
      geom_point(data = test2_df,
                 aes(x = y_test2, y = y_test2_pred),
                 col = "#89CFF0")
    p + labs(x = "y true", y = "y pred")
  }

## Load Data
path_base <- "~/Documents/PhD/02Research/01Papers/03Nullspace/"
path <- paste(path_base, "HDFeat/data/lfp_slim.csv", sep = "")

lfp_data = import(path)
# head(lfp_data)

## Construct Data Matrices
train_id <- lfp_data[, 1004] == 0
test1_id <- lfp_data[, 1004] == 1
test1_id[43] <- F # Removing the shortest lived outlier battery!
test2_id <- lfp_data[, 1004] == 2
y_mean <-
  import(paste(path_base, "HDFeat/data/lfp_y_mean.csv", sep = ""))
y_cm <-
  import(paste(path_base, "HDFeat/data/lfp_y_cm.csv", sep = ""))

X <- unname(as.matrix(rev(lfp_data[, 2:1001])))
X_list <- center_X_train(X)
X_ = X_list$X_

X_train <- X[train_id, ]
X_test1 <- X[test1_id, ]
X_test2 <- X[test2_id, ]

X_train_list <- center_X_train(X_train)
X_train_ <- X_train_list$X_
X_test1_ <- center_X_test(X_test1, X_train_list)
X_test2_ <- center_X_test(X_test2, X_train_list)


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
  coef(cvfit, s = "lambda.1se", excact = T)[2:1001,],
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
# First we have to define the Matrix D
x <-
  c(rep(0, dim(X_train)[2] - 2), 1,-1, rep(0, dim(X_train)[2] - 1))
D_step <- toeplitz2(x, 1000, 1000)
D_step_sparse <- as(D_step, "sparseMatrix")
# Its an interplay of the scale of X and the epsilon that will be added as ridge penalty.
# Actually this is rather a smooth EN. Even though the authors did not make it explicit,
# by setting eps you can actually run a smoother version of the elastic net.
#. eps = 1e-4, work well for the data in the origfinal scale. If the data is rescaled,
# might be worrh to reconsinder how epsilon should be set.
# --> Killer for functional data.
fl <-
  fusedlasso(
    y_train_,
    X_train_,
    D_step_sparse,
    gamma = 0,
    minlam = 1e-7,
    eps = 1e-4,
    rtol = 1e-11,
    # btol = 1e-11
  )
#, eps = 0.1)#, minlam = 0.000001)
plot(fl)
coeff_fused_lasso = coef(fl, lambda = 0.005, exact = T)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")

x <-
  c(rep(0, dim(X_train)[2] - 3), 0.5,-1, 0.5, rep(0, dim(X_train)[2] - 1))
D_p2 <- toeplitz2(x, 1000, 1000)
D_p2_sparse <- as(D_p2, "sparseMatrix")
# Its important to make sure that the the rowsum is 1 // that the toeplitz matirx
# is rescaled. Need to build some more intuition aroud how this actually works. 
fl_p2 <-
  fusedlasso(y_train_,
           X_train_,
           D_p2_sparse,
           gamma = 0.1,
           minlam = 1e-7,
           )
plot(fl_p2)
coeff_fused_lasso = coef(fl_p2, lambda = 1)
plot(x_lfp, coeff_fused_lasso$beta, type = "l")

# x <- c(rep(0, dim(X_train)[2]-4), 1, -3, -3, 1, rep(0, dim(X_train)[2]-1))
# D_p3 <- toeplitz2(x, 1000, 1000)
# D_p3_sparse <- as(D_p3, "sparseMatrix")
# fl_p3 <-  genlasso(y_train_, X_train_, D_p3_sparse)



# Trend filtering is an alternative. Differences in solution to be verified.
# Should be equal to fusedlasso1d. However, takes a lot longer to compute
# Regression coefficients look different.
# tf1 = trendfilter(y_train_, X=X_train_, ord=0)
# plot(tf1)
# coeff_fused_lasso = coef(tf1, lambda=0.0001)
# plot(x_lfp, coeff_fused_lasso$beta, type = "l")

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

# Check the predictions. (put the prediction stuff in a function for easy calling!)
# Interesting coefficients!
lambda_val <- 0.005
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
# TBH: Pretty neat results. It struggles for the long lived cells, but that's well known.

lambda_val <- 0.0001
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


# Open Questions for Sunday:
# Estimate the stats with qut.
# Does this make sense? Maybe kick this out, added value minimal!

# Can the nullspace regression coefficients be reformulated as a lasso problem?



# CLEAN UP #################################################
# Clear environment
rm(list = ls())
# Clear packages
p_unload(all)  # Remove all add-ons
detach("package:datasets", unload = TRUE)  # For base
# Clear plots
dev.off()  # But only if there IS a plot
# Clear console
cat("\014")  # ctrl+L
# Clear mind :)
