#pragma once

namespace FTPRL {
  
/**
 * Abstract Type of Matrix Data
 */
template<typename IndexType, typename ItorType>
class Matrix {

protected:
  IndexType ncol, nrow;

public:

  Matrix(IndexType _ncol, IndexType _nrow) :
  ncol(_ncol), nrow(_nrow) { }
  
  virtual ~Matrix() { }
  
  inline IndexType nfeature() const {
    return ncol;
  }
  
  inline IndexType ninstance() const {
    return nrow;
  }
  
  virtual ItorType getFeatureItorBegin(IndexType instance_id) const = 0;
  
  virtual ItorType getFeatureItorEnd(IndexType instance_id) const = 0;
  
  virtual IndexType getFeatureId(ItorType feature_iterator) const = 0;
  
  virtual double getValue(ItorType feature_iterator) const = 0;
  
};

}