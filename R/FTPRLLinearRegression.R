FTPRLLinearRegression <- "FTPRLLinearRegression"

#'@exportClass FTPRLLinearRegression
#'@title Prameters of Logistic Regression with FTPRL
#'@name FTPRLLinearRegression
#'@seealso \link{FTPRL}
setClass(FTPRLLinearRegression, representation(z = "numeric", n = "numeric"), contains = "FTPRL")

#'@export
init_FTPRLLinearRegression <- function(alpha, beta, lambda1, lambda2, nfeature) {
  .obj <- new("FTPRLLinearRegression")
  .obj@alpha <- alpha
  .obj@beta <- beta
  .obj@lambda1 <- lambda1
  .obj@lambda2 <- lambda2
  .obj@z <- numeric(nfeature)
  .obj@n <- numeric(nfeature)
  .obj
}

#'@export
update.FTPRLLinearRegression <- function(learner, data, y) {
  update_FTPRLLinearRegression(data, y, learner)
}

#'@export
predict.FTPRLLinearRegression <- function(learner, data) {
  predict_FTPRLLinearRegression(data, learner)
}

#'@export
update_FTPRLLinearRegression <- function(data, y, learner) {
  UseMethod("update_FTPRLLinearRegression")
}

#'@export
predict_FTPRLLinearRegression <- function(data, learner) {
  UseMethod("predict_FTPRLLinearRegression")
}