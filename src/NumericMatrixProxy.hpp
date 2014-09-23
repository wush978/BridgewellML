#pragma once

#include <Rcpp.h>
#include "Matrix.hpp"

class NumericMatrixProxy : public FTPRL::Matrix<int, size_t> {
  
  typedef int IndexType;
  
  typedef size_t ItorType;
  
  const double *data;
  
public:
  
  NumericMatrixProxy(Rcpp::NumericMatrix src)
  : Matrix<IndexType, ItorType>(src.ncol(), src.nrow()), data(REAL(Rcpp::wrap(src)))
  { }
  
  virtual ~NumericMatrixProxy() { }
  
  virtual ItorType getFeatureItorBegin(IndexType instance_id) const {
    return instance_id * nfeature();
  }
  
  virtual ItorType getFeatureItorEnd(IndexType instance_id) const {
    return (instance_id + 1) * nfeature();
  }
  
  virtual IndexType getFeatureId(ItorType feature_iterator) const {
    return feature_iterator % nfeature();
  }
  
  virtual double getValue(ItorType feature_iterator) const {
    int instance_id = feature_iterator / nfeature();
    int feature_id = feature_iterator % nfeature();
    return data[feature_id * ninstance() + instance_id];
  }
  
};