# A couple of functions that are handy for testing the fused lasso and co
# (Probably this implementation is somewhat naive as I'm not an R native)
# Author and Copyright: Joachim Schaeffer, joachim.schaeffer@posteo.de

mean_sd <- function(X) {
  mean_ <- colMeans(X)
  # TODO: TEST! Issue wiuth standard deviation!!
  sd <- sqrt(drop(scale(X, mean_, FALSE) ^ 2))
  return <- list(mean = mean_, sd = sd)
}

# EXPERIMENTAL FUNCTION
mean_sd_fused_lasso <- function (x, weights = rep(1, nrow(x)))
{
  weights <- weights / sum(weights)
  xm <- drop(t(weights) %*% x)
  xv <- drop(t(weights) %*% scale(x, xm, FALSE) ^ 2)
  xv[xv < 10 * .Machine$double.eps] <- 0
  list(mean = xm, sd = sqrt(xv))
}

centerXtrain <- function(X) {
  mean_ <- colMeans(X)
  X_ <- sweep(X, 2, mean_, "-")
  return <- list("X_" = X_, "mean" = mean_)
}

centerXtest <- function(X, X_train_list) {
  X_ <- sweep(X, 2, X_train_list$mean, "-")
  return <-  X_
}

standardize_y_train <- function(y) {
  N <- dim(y)[1]
  if (is.null(N)) {
    N <- size(y)[2]
  }
  mean_y <- mean(y)
  std_y <- sqrt(var(y) * (N - 1) / N) [1, 1]
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

plot_predictions_lfp <-
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

plot_one_set_predictions <-
  function(y_train,
           y_pred,
           y_train_list) {
    # Well I guess this is best to do in ggplot!
    train_df <-
      setNames(data.frame(y_train, rescale_y(y_pred, y_train_list)),
               c("y_train", "y_train_pred"))
    err_train = rmserr(train_df$y_train, train_df$y_train_pred)
    print(err_train$rmse)

    p <- ggplot() +
      geom_point(data = train_df,
                 aes(x = y_train, y = y_train_pred),
                 color = "#cc0000") +
      geom_abline(intercept = 0, slope = 1)
    p + labs(x = "y true", y = "y pred")
  }

predict_plot_lfp <-
  function(trained_model,
           lambda_val,
           X_train_,
           X_test1_,
           X_test2_,
           y_train_list,
           model = "cvfit") {
    if (model == "cvfit") {
      y_pred <-
        predict(trained_model, X_train_, s = lambda_val)
      y_pred_test1 <-
        predict(trained_model, X_test1_, s = lambda_val)
      y_pred_test2 <-
        predict(trained_model, X_test2_, s = lambda_val)
    }
    else {
      y_pred <-
        predict(trained_model,
                lambda = lambda_val,
                Xnew = X_train_)$fit
      y_pred_test1 <-
        predict(trained_model,
                lambda = lambda_val,
                Xnew = X_test1_)$fit
      y_pred_test2 <-
        predict(trained_model,
                lambda = lambda_val,
                Xnew = X_test2_)$fit
    }
    plot_predictions_lfp(y_train,
                         y_pred,
                         y_test1,
                         y_pred_test1,
                         y_test2,
                         y_pred_test2,
                         y_train_list)
  }

cv_genlasso <-
  function(X_train_,
           y_train_,
           D,
           nfolds = 5,
           plot_cv = T,
           minlam = c(1e-9, 1e-9),
           maxsteps = c(2000, 2000)) {
    N <- nrow(X_train_)
    foldid <- sample(rep(seq(nfolds), length = N))
    op <- options(nwarnings = 10000) # Keep all warnings!
    genlasso.fit <-
      genlasso(y_train_,
               X_train_,
               D,
               maxsteps = maxsteps[1],
               minlam = minlam[1])
    ## Evaluate each lambda on each fold:
    fold.lambda.losses <-
      tapply(seq_along(foldid), foldid, function(fold.indices) {
        disp("step")
        fold.genlasso.fit <- genlasso(
          y = y_train_[-fold.indices],
          X = X_train_[-fold.indices,],
          D = D,
          maxsteps = maxsteps[2],
          minlam = minlam[2],
        )

        fold.genlasso.preds <- predict(fold.genlasso.fit,
                                       lambda = genlasso.fit$lambda,
                                       Xnew = X_train_[fold.indices, ])$fit

        lambda.losses <-
          colMeans((fold.genlasso.preds - y_train_[fold.indices]) ^ 2)

        return (lambda.losses)
      })
    # CV loss for each lambda:
    cv.lossmatrix <- do.call(rbind, fold.lambda.losses)
    cv.lambda.losses <- colMeans(cv.lossmatrix)
    cv.lambda.losses_sd <- colStdevs(cv.lossmatrix)
    cv.genlasso.lambda.min <-
      genlasso.fit$lambda[which.min(cv.lambda.losses)]

    id_min <-
      which(cv.lambda.losses == min(cv.lambda.losses), arr.ind = TRUE)
    lambda_min_loss <-
      genlasso.fit$lambda[id_min]

    print("Min Lambda CV")
    print(lambda_min_loss)

    loss_larger_than_1se <-
      cv.lambda.losses > (cv.lambda.losses_sd[id_min] + min(cv.lambda.losses))
    id_1se <- min(which(loss_larger_than_1se == FALSE))
    lambda_cv_1se <- genlasso.fit$lambda[id_1se]
    print("Min Lambda 1SE CV")
    print(lambda_cv_1se)
    if (plot_cv == TRUE) {

    }
    matplot(
      genlasso.fit$lambda,
      cv.lambda.losses,
      type = "l",
      ylab = "loss",
      xlab = "lambda",
      main = "CV",
      log = "x"
    )
    matplot(
      genlasso.fit$lambda,
      cv.lambda.losses + cv.lambda.losses_sd,
      type = "l",
      ylab = "loss",
      xlab = "lambda",
      main = "CV",
      log = "x",
      add = TRUE
    )
    return <-
      list(
        "genlasso.fit" = genlasso.fit,
        "lossmatrix" = cv.lossmatrix,
        "lambda_vals" = genlasso.fit$lambda,
        "lambda.min" = lambda_min_loss,
        "lambda.1se" = lambda_cv_1se
      )
  }
