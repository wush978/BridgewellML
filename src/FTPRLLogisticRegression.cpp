#include <memory>
#include <Rcpp.h>
#include "NumericMatrixProxy.hpp"
#include "FTPRLProxy.hpp"
#include "LogisticRegression.hpp"

using namespace Rcpp;

//'@export
// [[Rcpp::export("update_FTPRLLogisticRegression.matrix")]]
void update_FTPRLLogisticRegression_matrix(NumericMatrix Rm, LogicalVector y, S4 Rlearner) {
  if (Rm.nrow() != y.size()) throw std::invalid_argument("");
  std::shared_ptr<NumericMatrixProxy> m(new NumericMatrixProxy(Rm));
  std::shared_ptr<FTPRLProxy> learner(new FTPRLProxy(Rlearner));
  typedef FTPRL::LogisticRegression<int> LR;
  double *z = REAL(Rlearner.slot("z")), *n = REAL(Rlearner.slot("n"));
  std::shared_ptr<LR>(new LR(learner, m->getNFeature(), z, n))->update<size_t, int>(m.get(), LOGICAL(y));
}

//'@export
// [[Rcpp::export("predict_FTPRLLogisticRegression.matrix")]]
SEXP predict_FTPRLLogisticRegression_matrix(NumericMatrix Rm, S4 Rlearner) {
  std::shared_ptr<NumericMatrixProxy> m(new NumericMatrixProxy(Rm));
  std::shared_ptr<FTPRLProxy> learner(new FTPRLProxy(Rlearner));
  typedef FTPRL::LogisticRegression<int> LR;
  double *z = REAL(Rlearner.slot("z")), *n = REAL(Rlearner.slot("n"));
  NumericVector retval(m->getNInstance(), 0.0);
  double *pretval = REAL(wrap(retval));
  std::shared_ptr<LR>(new LR(learner, m->getNFeature(), z, n))->predict<size_t>(m.get(), pretval);
  return retval;
}