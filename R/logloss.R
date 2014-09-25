#'@export
#'@title Log Loss Evaluator
#'@param p numeric vector. The predicted probability
#'@param y logical vector. The answer
#'@param tol numeric value. The value larger than $1 - tol$ will be shrinkage to $1 - tol$. 
#'The value smaller than $tol$ will be shrinkage to $tol$. 
logloss <- function(p, y, tol = 1e-6) {
  p[p > 1-tol] <- 1-tol
  p[p < tol] <- tol
  - y * log(p) - (1 - y) * log(1 - p)
}