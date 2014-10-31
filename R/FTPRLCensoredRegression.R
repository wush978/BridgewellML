FTPRLCensoredRegression <- "FTPRLCensoredRegression"

#'@exportClass FTPRLCensoredRegression
#'@title Prameters of Logistic Regression with FTPRL
#'@name FTPRLCensoredRegression
#'@seealso \link{FTPRL}
setClass(FTPRLCensoredRegression, representation(z = "numeric", n = "numeric"), contains = "FTPRL")

#'@export
init_FTPRLCensoredRegression <- function(alpha, beta, lambda1, lambda2, nfeature) {
  .obj <- new("FTPRLCensoredRegression")
  .obj@alpha <- alpha
  .obj@beta <- beta
  .obj@lambda1 <- lambda1
  .obj@lambda2 <- lambda2
  .obj@z <- numeric(nfeature)
  .obj@n <- numeric(nfeature)
  .obj
}

#'@export
update.FTPRLCensoredRegression <- function(learner, data, y, is_observed) {
  update_FTPRLCensoredRegression(data, y, is_observed, learner)
}

#'@export
predict.FTPRLCensoredRegression <- function(learner, data) {
  predict_FTPRLCensoredRegression(data, learner)
}

#'@export
update_FTPRLCensoredRegression <- function(data, y, is_observed, learner) {
  UseMethod("update_FTPRLCensoredRegression")
}

#'@export
predict_FTPRLCensoredRegression <- function(data, learner) {
  UseMethod("predict_FTPRLCensoredRegression")
}