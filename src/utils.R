# A couple of functions that are handy for testing the fused lasso and co
# (Probably this implementation is somewhat naive as I'm not an R native)
# Author and Copyright: Joachim Schaeffer, joachim.schaeffer@posteo.de

mean_sd <- function(X) {
  mean_ <- colMeans(X)
  # TODO: TEST! Issue wiuth standard deviation!!
  sd <- sqrt(drop(scale(X, mean_, FALSE)^2))
  return <- list(mean = mean_, sd = sd)
}

mean_sd_fused_lasso <- function (x, weights = rep(1, nrow(x))) 
{
  weights <- weights/sum(weights)
  xm <- drop(t(weights) %*% x)
  xv <- drop(t(weights) %*% scale(x, xm, FALSE)^2)
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
  N <-dim(y)[1]
  if (is.null(N)){
    N <- size(y)[2]
  }
  mean_y <- mean(y)
  std_y <- sqrt(var(y) * (N - 1) / N) #[1, 1]
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