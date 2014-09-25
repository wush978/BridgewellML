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
  std::shared_ptr<LR>(new LR(learner, m->nfeature(), z, n))->update<size_t, int>(m.get(), LOGICAL(y));
  
}
