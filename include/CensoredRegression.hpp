#pragma once

#include <memory>
#include "Matrix.hpp"
#include "FTPRL.hpp"

extern "C" {

  // standard normal pdf
  double dnorm(double);
  
  // standard normal cdf
  double pnorm(double);

}

namespace FTPRL {

template<typename IndexType>
class CensoredRegression {
  
  bool is_manage_memory;
  
  double *z, *n;
  
  FTPRL* ftprl;

public:
  
  CensoredRegression(FTPRL* _ftprl, IndexType nfeature) 
  : ftprl(_ftprl), z(new double[nfeature]), n(new double[nfeature]), is_manage_memory(true)
  { }
  
  CensoredRegression(FTPRL* _ftprl, IndexType nfeature, double *_z, double *_n) 
  : ftprl(_ftprl), z(_z), n(_n), is_manage_memory(false) 
  { } 
  
  virtual ~CensoredRegression() {
    if (is_manage_memory) {
      delete [] z;
      delete [] n;
    }
  }
  
  template<typename ItorType, typename ValueType, typename LabelType>
  void update(Matrix<IndexType, ItorType>* m, ValueType* y, LabelType* is_observed, bool is_skip(ValueType, LabelType)) {
    // The last parameter is log(sigma)
    const Matrix<IndexType, ItorType>::IndexType nfeature = m->getNFeature();
    for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
      if (is_skip(y[instance_id], is_observed[instance_id])) continue;
      double pred = 0, g0 = 0, sigma = exp(ftprl->get_w(z[nfeature], n[nfeature]));
      #pragma omp parallel for reduction(+:pred)
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double w = ftprl->get_w(z[feature_id], n[feature_id]);
        pred += value * w;
      }
      double zscore = (pred - y) / sigma;
      double dzscore_dloss = (is_observed[instance_id] ? zscore : - dnorm(zscore) / pnorm(zscore));
      #pragma omp parallel for
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double g = dzscore_dloss * value / sigma;
        ftprl->update_zn(g, z + feature_id, n + feature_id);
      }
      { // update sigma
        IndexType &feature_id(nfeature);
        double g = - dzscore_dloss * zscore;
        ftprl->update_zn(g, z + feature_id, n + feature_id);
      }
    }
  }

  template<typename ItorType>
  void predict(Matrix<IndexType, ItorType>* m, double* y) {
    #pragma omp parallel for
    for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
      double& pred(y[instance_id]);
      pred = 0;
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter != m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double w = ftprl->get_w(z[feature_id], n[feature_id]);
        pred += value * w;
      }
    }
  }

};
  
}