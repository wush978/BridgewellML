#pragma once

#include "Matrix.hpp"

namespace FTPRL {

template<typename IndexType>
class DenseMatrix : public Matrix<IndexType> {

public:

  DenseMatrix(IndexType _ncol, IndexType _nrow)
  : Matrix<IndexType>(_ncol, _nrow)
  { }
  
  virtual ~DenseMatrix() { }
  
  virtual IndexType nfeature() const {
    return Matrix<IndexType>::ncol;
  }
  
  virtual IndexType ninstance() const {
    return Matrix<IndexType>::nrow;
  }
  
  virtual IndexType getFeatureItorBegin(IndexType instance_id) const {
    return instance_id * Matrix<IndexType>::ncol;
  }
  
  virtual IndexType getFeatureItorEnd(IndexType instance_id) const {
    return (instance_id + 1) * Matrix<IndexType>::ncol;
  }
  
  virtual IndexType getFeatureId(IndexType feature_iterator) const {
    return feature_iterator % Matrix<IndexType>::ncol;
  }
  
  virtual double getValue(IndexType feature_iterator) const = 0;
  
};
  
}