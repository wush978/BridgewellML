#pragma once

#include <Rcpp.h>
#include "DenseMatrix.hpp"

class NumericMatrixProxy : public FTPRL::DenseMatrix<int> {
  
  const double *data;
  
public:
  
  NumericMatrixProxy(Rcpp::NumericMatrix src)
  : DenseMatrix<int>(src.ncol(), src.nrow()), data(REAL(Rcpp::wrap(src)))
  { }
  
  virtual ~NumericMatrixProxy() { }
  
  virtual double getValue(int feature_iterator) const {
    int instance_id = feature_iterator / FTPRL::Matrix<int>::ncol;
    int feature_id = feature_iterator % FTPRL::Matrix<int>::ncol;
    return data[feature_id * nrow + instance_id];
  }
  
};