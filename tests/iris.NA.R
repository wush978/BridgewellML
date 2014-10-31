library(BridgewellML)
library(FeatureHashing)
m <- hashed.model.matrix(Sepal.Length ~ ., iris, 2^4)
y <- iris$Sepal.Length
y[sample(seq_along(y), 10, FALSE)] <- NA
learner <- init_FTPRLLinearRegression(0.1, 1, 0.1, 0.1, nrow(m))
update_FTPRLLinearRegression(m, y, learner)
stopifnot(sum(is.na(learner@z)) == 0)
