FTPRLNeuronNetwork <- "FTPRLNeuronNetwork"

#'@exportClass FTPRLNeuronNetwork
#'@title Prameters of Logistic Regression with FTPRL
#'@name FTPRLNeuronNetwork
#'@seealso \link{FTPRL}
setClass(FTPRLNeuronNetwork, representation(z = "list", n = "list", nnode = "integer"), contains = "FTPRL")

#'@export
init_FTPRLNeuronNetwork <- function(alpha, beta, lambda1, lambda2, nnode) {
  stopifnot(length(nnode) >= 3)
  .obj <- new("FTPRLNeuronNetwork")
  .obj@alpha <- alpha
  .obj@beta <- beta
  .obj@lambda1 <- lambda1
  .obj@lambda2 <- lambda2
  .obj@nnode <- nnode
  .obj@z <- vector("list", length(nnode) - 1)
  .obj@n <- vector("list", length(nnode) - 1)
  for(i in head(seq_along(nnode), -1)) {
    .obj@z[[i]] <- numeric(nnode[i] * nnode[i + 1])
    .obj@n[[i]] <- numeric(nnode[i] * nnode[i + 1])
  }
  .obj
}

#'@export
update.FTPRLNeuronNetwork <- function(learner, data, y) {
  update_FTPRLNeuronNetwork(data, y, learner)
}

#'@export
predict.FTPRLNeuronNetwork <- function(learner, data) {
  predict_FTPRLNeuronNetwork(data, learner)
}

#'@export
update_FTPRLNeuronNetwork <- function(data, y, learner) {
  UseMethod("update_FTPRLNeuronNetwork")
}

#'@export
predict_FTPRLNeuronNetwork <- function(data, learner) {
  UseMethod("predict_FTPRLNeuronNetwork")
}