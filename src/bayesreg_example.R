# Author: Joachim Schaeffer, 2023, joachim.schaeffer@posteo.de
library(MASS)
library(bayesreg)

rm(list = ls())
n <- 10 # sample size
p <- 50 # number of predictors

rho <- 0.5

S <- toeplitz(rho^(0:(p - 1)))
print(S)
X <- mvrnorm(n, rep(0, p), S)


b <- as.vector(c(5, 3, 3, 1, 1, rep(0, p - 5)))
snr <- 4
mu <- X %*% b
s2 <- var(mu) / snr
y <- mu + rnorm(n, 0, sqrt(s2))

df <- data.frame(y = y, X = X)

# plot all rows of the dataframe
matplot(t(data.frame(X)), type = "l")

# Cool! Now let's try to fit a Bayesian linear regression model to this data.
# We'll use the bayesreg package for this.
rv <- bayesreg(
    y ~ .,
    data = df,
    model = "gaussian",
    prior = "ridge",
    n.samples = 1e4,
    burnin = 1000,
    thin = 1
    )

rv.s <- summary(rv)
