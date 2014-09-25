#pragma once

#include <memory>
#include "Matrix.hpp"
#include "FTPRL.hpp"

namespace FTPRL {

template<typename IndexType>
class LogisticRegression {
  
  bool is_manage_memory;
  
  double *z, *n;
  
  std::shared_ptr<FTPRL> ftprl;

public:
  
  LogisticRegression(std::shared_ptr<FTPRL> _ftprl, IndexType nfeature) 
  : ftprl(_ftprl), z(new double[nfeature]), n(new double[nfeature]), is_manage_memory(true)
  { }
  
  LogisticRegression(std::shared_ptr<FTPRL> _ftprl, IndexType nfeature, double *_z, double *_n) 
  : ftprl(_ftprl), z(_z), n(_n), is_manage_memory(false) 
  { } 
  
  virtual ~LogisticRegression() {
    if (is_manage_memory) {
      delete [] z;
      delete [] n;
    }
  }
  
  template<typename ItorType, typename LabelType>
  void update(Matrix<IndexType, ItorType>* m, LabelType* y) {
    for(IndexType instance_id = 0;instance_id < m->ninstance();instance_id++) {
      double pred = 0, g0 = 0;
      #pragma omp parallel for
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter != m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double w = ftprl->get_w(z[feature_id], n[feature_id]);
        pred += value * w;
      }
      pred = sigma(pred);
      g0 = pred - y[instance_id];
      #pragma omp parallel for
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter != m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double g = value * g0;
        ftprl->update_zn(g, z + feature_id, n + feature_id);
      }
    }
  }

  template<typename ItorType>
  void predict(Matrix<IndexType, ItorType>* m, double* y) {
    #pragma omp parallel for
    for(IndexType instance_id = 0;instance_id < m->ninstance();instance_id++) {
      double& pred(y[instance_id]);
      pred = 0;
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter != m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double w = ftprl->get_w(z[feature_id], n[feature_id]);
        pred += value * w;
      }
      pred = sigma(pred);
    }
  }

  inline static double sigma(double x) {
    return 1 / (1 + std::exp(-x));
  }
  
};
  
}