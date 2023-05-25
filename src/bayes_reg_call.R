# Interface to bayesreg can be called from a python script.
# It will take the data as input and return the output of the bayesreg model.
source("src/bayes_reg_local_code.R")

library(MASS)
library(bayesreg)

bayes_reg_call <- function(X, y, samples, burnin, thin) {
    df <- data.frame(y = y, X = X)
    result <- bayesreg(
    y ~ .,
    data = df,
    model = "gaussian",
    prior = "ridge",
    n.samples = samples,
    burnin = burnin,
    thin = thin
    )
    result
}

# Load the data from the csv file
lfp_data <- read.table("data/lfp_slim.csv", header = TRUE, sep = ",")
train_id <- lfp_data[, 1004] == 0
y_mean <- read.table("data/lfp_y_mean.csv", header = TRUE, sep = ",")
y_cm <- read.table("data/lfp_y_cm.csv", header = TRUE, sep = ",")

# Convert the data into a matrix x and a vector y
X <- unname(as.matrix(lfp_data[, 2:1001]))

X_train <- X[train_id, ]
y_train_mean <- rowMeans(X_train)
# Subtract the column mean from the matrix X
X_train_ <- sweep(X_train, 2, colMeans(X_train), "-")
y_train_mean_ <- y_train_mean - mean(y_train_mean)

df_train <- data.frame(y = y_train_mean_, X = X_train)

if (0){
    rv <- bayesreg(
        y_train_mean_ ~ .,
        data = df_train,
        model = "gaussian",
        prior = "ridge",
        n.samples = 1e4,
        burnin = 1000,
        thin = 1
        )
    rv.s <- summary(rv)
    write.table(rv.s, file = "data/r/rv.csv", sep = ",")
}


rv_local <- bayesreg_local(
    y_train_mean_ ~ .,
    data = df_train,
    model = "gaussian",
    prior = "ridge",
    n.samples = 1e4,
    burnin = 1000,
    thin = 1,
    std = FALSE
    )
rv_local.s <- summary(rv_local)
write.table(rv_local.s, file = "data/r/rv_local.csv", sep = ",")
