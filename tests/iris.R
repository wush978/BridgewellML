library(FTPRL)
m <- model.matrix(Species ~ ., iris)
learner <- init_FTPRLLogisticRegression(0.1, 1, 0.1, 0.1, ncol(m))
update_FTPRLLogisticRegression(m, iris$Species == "setosa", learner)
learner@z
learner@n
