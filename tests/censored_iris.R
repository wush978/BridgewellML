library(BridgewellML)
library(Matrix)
m <- sparse.model.matrix(Sepal.Length ~ ., iris, transpose = TRUE)
y <- iris$Sepal.Length
is_observed <- rep(TRUE, 150)
is_observed[sample(1:150, 30, FALSE)] <- FALSE
y[!is_observed] <- y[!is_observed] - 1
learner <- init_FTPRLCensoredRegression(1, 1, 0.1, 0.1, nrow(m) + 1)
counter <- 0
for(.i in 1:100) {
  update_FTPRLCensoredRegression(m, y, is_observed, learner)
  y.fitted <- predict_FTPRLCensoredRegression(m, learner)
  sigma <- exp(tail(learner$w, 1))
  if (counter == 1000) {
    print(sum(
      dnorm(y[is_observed], y.fitted[is_observed], sigma, log = TRUE),
      pnorm(y[!is_observed], y.fitted[!is_observed], sigma, lower.tail = FALSE, log.p = TRUE)))
    counter <- 0
  } else {
    counter <- counter + 1
  }
}

