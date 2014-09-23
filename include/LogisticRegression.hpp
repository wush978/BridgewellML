#pragma once

#include <memory>
#include "Matrix.hpp"
#include "FTPRL.hpp"

namespace FTPRL {

template<typename IndexType>
class LogisticRegression {
  
  double *z, *n;
  
  std::shared_ptr<FTPRL> ftprl;

public:
  
  LogisticRegression(std::shared_ptr<FTPRL> _ftprl, IndexType nfeature) 
  : ftprl(_ftprl), z(new double[nfeature]), n(new double[nfeature])
  { }
  
  virtual ~LogisticRegression() {
    delete [] z;
    delete [] n;
  }
  
  template<typename IndexType, typename ItorType>
  void update(Matrix<IndexType, ItorType>* m, bool* y) {
    for(IndexType instance_id = 0;instance_id < m->ninstance;instance_id++) {
      double pred = 0, g0 = 0;
      #pragma omp parallel for
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter != m->getFeatureItorEnd(instance_id), iter++) {
        IndexType feature_d = getFeatureId(iter);
        double value = getValue(iter);
        double w = ftrl->get_w(z[feature_id], n[feature_id]);
        pred += value * w;
      }
      pred = sigma(pred);
      g0 = pred - y[instance_id];
      #pragma omp parallel for
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter != m->getFeatureItorEnd(instance_id), iter++) {
        IndexType feature_id = getFeatureId(iter);
        double value = getValue(iter);
        double g = value * g0;
        update_zn(g, z + feature_id, n + feature_id);
      }
    }
    
  }
  
  inline static double sigma(double x) {
    return 1 / (1 + std::exp(-x));
  }
  
};
  
}