#include <Rcpp.h>
#include "FTPRLProxy.hpp"

using namespace Rcpp;

//[[Rcpp::export(".get_w")]]
SEXP get_w(S4 RLearner) {
  FTPRLProxy learner(RLearner);
  NumericVector z(RLearner.slot("z")), n(RLearner.slot("n"));
  NumericVector retval(z.size());
  #pragma omp parallel for
  for(size_t i = 0;i < z.size();i++) {
    retval[i] = learner.get_w(z[i], n[i]);
  }
  return retval;
}