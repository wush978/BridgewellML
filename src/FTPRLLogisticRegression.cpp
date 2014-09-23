#include <memory>
#include <Rcpp.h>
#include "NumericMatrixProxy.hpp"
#include "FTPRLProxy.hpp"
#include "LogisticRegression.hpp"

using namespace Rcpp;

//'@export
// [[Rcpp::export("update_FTPRLLogisticRegression.matrix")]]
void update_FTPRLLogisticRegression_matrix(NumericMatrix Rm, S4 Rlearner) {
  std::shared_ptr<NumericMatrixProxy> m(new NumericMatrixProxy(Rm));
  std::shared_ptr<FTPRLProxy> learner(new FTPRLProxy(Rlearner));  
}
