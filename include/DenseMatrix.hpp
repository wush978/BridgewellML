#pragma once

#include "Matrix.hpp"

namespace FTPRL {

template<typename IndexType>
class DenseMatrix : public Matrix<IndexType> {

public:

  DenseMatrix(IndexType _ncol, IndexType _nrow)
  : Matrix(_ncol, _nrow)
  { }
  
  virtual ~DenseMatrix() { }
  
  virtual IndexType nfeature() const {
    return ncol;
  }
  
  virtual IndexType ninstance() const {
    return nrow;
  }
  
  virtual IndexType getFeatureItorBegin(IndexType instance_id) const {
    return instance_id * ncol;
  }
  
  virtual IndexType getFeatureItorEnd(IndexType instance_id) const {
    return (instance_id + 1) * ncol;
  }
  
  virtual IndexType getFeatureId(IndexType feature_iterator) const {
    return feature_iterator %% ncol;
  }
  
  virtual double getValue(IndexType feature_iterator) const = 0;
  
};
  
}