#pragma once

namespace FTPRL {
  
/**
 * Abstract Type of Matrix Data
 */
template<typename IndexType>
class Matrix {

protected:
  IndexType ncol, nrow;

public:

  Matrix(IndexType _ncol, IndexType _nrow) :
  ncol(_ncol), nrow(_nrow) { }
  
  virtual ~Matrix() { }
  
  virtual IndexType nfeature() const = 0;
  
  virtual IndexType ninstance() const = 0;
  
  virtual IndexType getFeatureItorBegin(IndexType instance_id) const = 0;
  
  virtual IndexType getFeatureItorEnd(IndexType instance_id) const = 0;
  
  virtual IndexType getFeatureId(IndexType feature_iterator) const = 0;
  
  virtual double getValue(IndexType feature_iterator) const = 0;
  
};

}