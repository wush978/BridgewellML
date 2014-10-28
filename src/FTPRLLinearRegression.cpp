#include <memory>
#include <Rcpp.h>
#include "NumericMatrixProxy.hpp"
#include "dgCMatrixProxy.hpp"
#include "FTPRLProxy.hpp"
#include "LinearRegression.hpp"

using namespace Rcpp;

template<typename InputType, typename MatrixProxy, typename IndexType>
void update_FTPRLLinearRegression(InputType Rm, NumericVector y, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> plearner(new FTPRLProxy(Rlearner));
  double *z = REAL(Rlearner.slot("z")), *n = REAL(Rlearner.slot("n"));
  FTPRL::LinearRegression<IndexType> lr(plearner.get(), m.getNFeature(), z, n);
  lr.update(&m, &y[0]);
}

//'@export
// [[Rcpp::export("update_FTPRLLinearRegression.matrix")]]
void update_FTPRLLinearRegression_matrix(NumericMatrix Rm, NumericVector y, S4 Rlearner) {
  if (Rm.nrow() != y.size()) throw std::invalid_argument("");
  update_FTPRLLinearRegression<NumericMatrix, NumericMatrixProxy, int>(Rm, y, Rlearner);
}

//'@export
// [[Rcpp::export("update_FTPRLLinearRegression.dgCMatrix")]]
void update_FTPRLLinearRegression_dgCMatrix(S4 Rm, NumericVector y, S4 Rlearner) {
  IntegerVector dim(Rm.slot("Dim"));
  if (dim[1] != y.size()) throw std::invalid_argument("");
  update_FTPRLLinearRegression<S4, dgCMatrixProxy, int>(Rm, y, Rlearner);
}

//'@export
// [[Rcpp::export("update_FTPRLLinearRegression.CSRMatrix")]]
void update_FTPRLLinearRegression_CSRMatrix(S4 Rm, NumericVector y, S4 Rlearner) {
  IntegerVector dim(Rm.slot("Dim"));
  if (dim[1] != y.size()) throw std::invalid_argument("");
  update_FTPRLLinearRegression<S4, dgCMatrixProxy, int>(Rm, y, Rlearner);
}

template<typename InputType, typename MatrixProxy, typename IndexType>
SEXP predict_FTPRLLinearRegression(InputType Rm, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> learner(new FTPRLProxy(Rlearner));
  double *z = REAL(Rlearner.slot("z")), *n = REAL(Rlearner.slot("n"));
  NumericVector retval(m.getNInstance(), 0.0);
  double *pretval = REAL(wrap(retval));
  FTPRL::LinearRegression<IndexType> lr(learner.get(), m.getNFeature(), z, n);
  lr.predict(&m, pretval);
  return retval;
}

//'@export
// [[Rcpp::export("predict_FTPRLLinearRegression.matrix")]]
SEXP predict_FTPRLLinearRegression_matrix(NumericMatrix Rm, S4 Rlearner) {
  return predict_FTPRLLinearRegression<NumericMatrix, NumericMatrixProxy, int>(Rm, Rlearner);
}

//'@export
// [[Rcpp::export("predict_FTPRLLinearRegression.dgCMatrix")]]
SEXP predict_FTPRLLinearRegression_dgCMatrix(S4 Rm, S4 Rlearner) {
  return predict_FTPRLLinearRegression<S4, dgCMatrixProxy, int>(Rm, Rlearner);
}

//'@export
// [[Rcpp::export("predict_FTPRLLinearRegression.CSRMatrix")]]
SEXP predict_FTPRLLinearRegression_CSRMatrix(S4 Rm, S4 Rlearner) {
  return predict_FTPRLLinearRegression<S4, dgCMatrixProxy, int>(Rm, Rlearner);
}
