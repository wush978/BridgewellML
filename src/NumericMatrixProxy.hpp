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
    return instance_id * getNFeature();
  }
  
  virtual ItorType getFeatureItorEnd(IndexType instance_id) const {
    return (instance_id + 1) * getNFeature();
  }
  
  virtual IndexType getFeatureId(ItorType feature_iterator) const {
    return feature_iterator % getNFeature();
  }
  
  virtual double getValue(ItorType feature_iterator) const {
    int instance_id = feature_iterator / getNFeature();
    int feature_id = feature_iterator % getNFeature();
    return data[feature_id * getNInstance() + instance_id];
  }
  
};