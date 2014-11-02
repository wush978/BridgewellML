#include <memory>
#include <Rmath.h>
#include <Rcpp.h>
#include "NumericMatrixProxy.hpp"
#include "dgCMatrixProxy.hpp"
#include "FTPRLProxy.hpp"
#include "CensoredRegression.hpp"

using namespace Rcpp;

double logdnorm(double x) {
  return ::Rf_dnorm4(x, 0, 1, 1);
}

double logpnorm(double x) {
  return ::Rf_pnorm5(x, 0, 1, 1, 1);
}

bool is_na(double value, int is_observed) {
  if (is_observed == NA_LOGICAL) return true;
  if (R_IsNA(value)) return true;
  return false;
}

template<typename InputType, typename MatrixProxy, typename IndexType>
void update_FTPRLCensoredRegression(InputType Rm, NumericVector y, LogicalVector is_observed, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> plearner(new FTPRLProxy(Rlearner));
  double *z = REAL(Rlearner.slot("z")), *n = REAL(Rlearner.slot("n"));
  FTPRL::CensoredRegression<IndexType> lr(plearner.get(), m.getNFeature(), z, n);
  lr.update(&m, &y[0], &is_observed[0], is_na);
}

//'@export
// [[Rcpp::export("update_FTPRLCensoredRegression.matrix")]]
void update_FTPRLCensoredRegression_matrix(NumericMatrix Rm, NumericVector y, LogicalVector is_observed, S4 Rlearner) {
  if (Rm.nrow() != y.size()) throw std::invalid_argument("");
  update_FTPRLCensoredRegression<NumericMatrix, NumericMatrixProxy, int>(Rm, y, is_observed, Rlearner);
}

//'@export
// [[Rcpp::export("update_FTPRLCensoredRegression.dgCMatrix")]]
void update_FTPRLCensoredRegression_dgCMatrix(S4 Rm, NumericVector y, LogicalVector is_observed, S4 Rlearner) {
  IntegerVector dim(Rm.slot("Dim"));
  if (dim[1] != y.size()) throw std::invalid_argument("");
  update_FTPRLCensoredRegression<S4, dgCMatrixProxy, int>(Rm, y, is_observed, Rlearner);
}

//'@export
// [[Rcpp::export("update_FTPRLCensoredRegression.CSRMatrix")]]
void update_FTPRLCensoredRegression_CSRMatrix(S4 Rm, NumericVector y, LogicalVector is_observed, S4 Rlearner) {
  IntegerVector dim(Rm.slot("Dim"));
  if (dim[1] != y.size()) throw std::invalid_argument("");
  update_FTPRLCensoredRegression<S4, dgCMatrixProxy, int>(Rm, y, is_observed, Rlearner);
}

template<typename InputType, typename MatrixProxy, typename IndexType>
SEXP predict_FTPRLCensoredRegression(InputType Rm, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> learner(new FTPRLProxy(Rlearner));
  double *z = REAL(Rlearner.slot("z")), *n = REAL(Rlearner.slot("n"));
  NumericVector retval(m.getNInstance(), 0.0);
  double *pretval = REAL(wrap(retval));
  FTPRL::CensoredRegression<IndexType> lr(learner.get(), m.getNFeature(), z, n);
  lr.predict(&m, pretval);
  return retval;
}

//'@export
// [[Rcpp::export("predict_FTPRLCensoredRegression.matrix")]]
SEXP predict_FTPRLCensoredRegression_matrix(NumericMatrix Rm, S4 Rlearner) {
  return predict_FTPRLCensoredRegression<NumericMatrix, NumericMatrixProxy, int>(Rm, Rlearner);
}

//'@export
// [[Rcpp::export("predict_FTPRLCensoredRegression.dgCMatrix")]]
SEXP predict_FTPRLCensoredRegression_dgCMatrix(S4 Rm, S4 Rlearner) {
  return predict_FTPRLCensoredRegression<S4, dgCMatrixProxy, int>(Rm, Rlearner);
}

//'@export
// [[Rcpp::export("predict_FTPRLCensoredRegression.CSRMatrix")]]
SEXP predict_FTPRLCensoredRegression_CSRMatrix(S4 Rm, S4 Rlearner) {
  return predict_FTPRLCensoredRegression<S4, dgCMatrixProxy, int>(Rm, Rlearner);
}
