library(BridgewellML)
m <- model.matrix(Species ~ ., iris)
y <- iris$Species == "setosa"
learner <- init_FTPRLLogisticRegression(0.1, 1, 0.1, 0.1, ncol(m))
r <- rep(0, 10)
for(i in 1:10) {
  update(learner, m, y)
  r[i] <- mean(logloss(predict(learner, m), y))
}
stopifnot(all(diff(r) < 0))
