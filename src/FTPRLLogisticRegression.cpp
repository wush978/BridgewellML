#include <memory>
#include <Rcpp.h>
#include "NumericMatrixProxy.hpp"
#include "dgCMatrixProxy.hpp"
#include "FTPRLProxy.hpp"
#include "LogisticRegression.hpp"

using namespace Rcpp;

template<typename InputType, typename MatrixProxy, typename IndexType, typename gggg>
void update_FTPRLLogisticRegression(InputType Rm, LogicalVector y, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> plearner(new FTPRLProxy(Rlearner));
  double *z = REAL(Rlearner.slot("z")), *n = REAL(Rlearner.slot("n"));
  FTPRL::LogisticRegression<IndexType> lr(plearner, m.getNFeature(), z, n);
  lr.update(&m, &y[0]);
}

//'@export
// [[Rcpp::export("update_FTPRLLogisticRegression.matrix")]]
void update_FTPRLLogisticRegression_matrix(NumericMatrix Rm, LogicalVector y, S4 Rlearner) {
  if (Rm.nrow() != y.size()) throw std::invalid_argument("");
  update_FTPRLLogisticRegression<NumericMatrix, NumericMatrixProxy, int, size_t>(Rm, y, Rlearner);
}

//'@export
// [[Rcpp::export("update_FTPRLLogisticRegression.dgCMatrix")]]
void update_FTPRLLogisticRegression_dgCMatrix(S4 Rm, LogicalVector y, S4 Rlearner) {
  IntegerVector dim(Rm.slot("Dim"));
  if (dim[1] != y.size()) throw std::invalid_argument("");
  update_FTPRLLogisticRegression<S4, dgCMatrixProxy, int, int>(Rm, y, Rlearner);
}

template<typename InputType, typename MatrixProxy, typename IndexType, typename ItorType>
SEXP predict_FTPRLLogisticRegression(InputType Rm, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> learner(new FTPRLProxy(Rlearner));
  double *z = REAL(Rlearner.slot("z")), *n = REAL(Rlearner.slot("n"));
  NumericVector retval(m.getNInstance(), 0.0);
  double *pretval = REAL(wrap(retval));
  FTPRL::LogisticRegression<IndexType> lr(learner, m.getNFeature(), z, n);
  lr.predict(&m, pretval);
  return retval;
}

//'@export
// [[Rcpp::export("predict_FTPRLLogisticRegression.matrix")]]
SEXP predict_FTPRLLogisticRegression_matrix(NumericMatrix Rm, S4 Rlearner) {
  return predict_FTPRLLogisticRegression<NumericMatrix, NumericMatrixProxy, int, size_t>(Rm, Rlearner);
}

//'@export
// [[Rcpp::export("predict_FTPRLLogisticRegression.dgCMatrix")]]
SEXP predict_FTPRLLogisticRegression_dgCMatrix(S4 Rm, S4 Rlearner) {
  return predict_FTPRLLogisticRegression<S4, dgCMatrixProxy, int, int>(Rm, Rlearner);
}
