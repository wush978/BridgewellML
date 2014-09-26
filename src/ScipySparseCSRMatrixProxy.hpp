#pragma once

#include "Matrix.hpp"

template<typename ValueType, typename IndexType, typename ItorType>
class ScipySparseCSRMatrixProxy : FTPRL::Matrix<IndexType, ItorType> {

  IndexType *i, *p;
  
  ValueType *x;
  
public:

  ScipySparseCSRMatrixProxy(IndexType _nfeature, IndexType _ninstance, IndexType *_i, IndexType *_p, ValueType *_x)
  : FTPRL::Matrix<IndexType, ItorType>(_nfeature, _ninstance), i(_i), p(_p), x(_x)
  { }
  
  virtual ~ScipySparseCSRMatrixProxy() { }
  
  virtual ItorType getFeatureItorBegin(IndexType instance_id) const {
    return p[instance_id];
  }
  
  virtual ItorType getFeatureItorEnd(IndexType instance_id) const {
    return p[instance_id + 1];
  }
  
  virtual IndexType getFeatureId(ItorType feature_iterator) const {
    return i[feature_iterator];
  }
  
  virtual double getValue(ItorType feature_iterator) const {
    return x[feature_iterator];
  }
  
};