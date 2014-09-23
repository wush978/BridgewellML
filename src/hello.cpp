#include <memory>
#include <Rcpp.h>
#include "NumericMatrixProxy.hpp"
using namespace Rcpp;

// [[Rcpp::export]]
void hello(NumericMatrix src) {
  std::shared_ptr<NumericMatrixProxy> srcp(new NumericMatrixProxy(src));
}
