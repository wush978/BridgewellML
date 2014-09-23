FTPRLLogisticRegression <- "FTPRLLogisticRegression"

#'@exportClass FTPRLLogisticRegression
setClass(FTPRLLogisticRegression, representation(z = "numeric", n = "numeric"), contains = "FTPRL")

#'@export
init_FTPRLLogisticRegression <- function(alpha, beta, lambda1, lambda2, nfeature) {
  .obj <- new("FTPRLLogisticRegression")
  .obj@alpha <- alpha
  .obj@beta <- beta
  .obj@lambda1 <- lambda1
  .obj@lambda2 <- lambda2
  .obj@z <- numeric(nfeature)
  .obj@n <- numeric(nfeature)
  .obj
}

#'@export
update_FTPRLLogisticRegression <- function(data, learner) {
  UseMethod("update_FTPRLLogisticRegression")
}