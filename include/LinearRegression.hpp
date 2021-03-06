#pragma once

#include <memory>
#include "Matrix.hpp"
#include "FTPRL.hpp"

namespace FTPRL {

template<typename IndexType>
class LinearRegression {
  
  bool is_manage_memory;
  
  double *z, *n;
  
  FTPRL* ftprl;

public:
  
  LinearRegression(FTPRL* _ftprl, IndexType nfeature) 
  : ftprl(_ftprl), z(new double[nfeature]), n(new double[nfeature]), is_manage_memory(true)
  { }
  
  LinearRegression(FTPRL* _ftprl, IndexType nfeature, double *_z, double *_n) 
  : ftprl(_ftprl), z(_z), n(_n), is_manage_memory(false) 
  { } 
  
  virtual ~LinearRegression() {
    if (is_manage_memory) {
      delete [] z;
      delete [] n;
    }
  }
  
  template<typename ItorType, typename ValueType>
  void update(Matrix<IndexType, ItorType>* m, ValueType* y) {
    for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
      double pred = 0, g0 = 0;
      #pragma omp parallel for reduction(+:pred)
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double w = ftprl->get_w(z[feature_id], n[feature_id]);
        pred += value * w;
      }
      g0 = 2 * (pred - y[instance_id]);
      #pragma omp parallel for
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double g = value * g0;
        ftprl->update_zn(g, z + feature_id, n + feature_id);
      }
    }
  }

  template<typename ItorType, typename ValueType>
  void update(Matrix<IndexType, ItorType>* m, ValueType* y, ValueType skip_value) {
    for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
      if (y[instance_id] == skip_value) continue;
      double pred = 0, g0 = 0;
      #pragma omp parallel for reduction(+:pred)
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double w = ftprl->get_w(z[feature_id], n[feature_id]);
        pred += value * w;
      }
      g0 = 2 * (pred - y[instance_id]);
      #pragma omp parallel for
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double g = value * g0;
        ftprl->update_zn(g, z + feature_id, n + feature_id);
      }
    }
  }
  
  template<typename ItorType, typename ValueType>
  void update(Matrix<IndexType, ItorType>* m, ValueType* y, bool is_skip(ValueType) ) {
    for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
      if (is_skip(y[instance_id])) continue;
      double pred = 0, g0 = 0;
      #pragma omp parallel for reduction(+:pred)
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id); iter++) {
        IndexType feature_id = m->getFeatureId(iter);
        double value = m->getValue(iter);
        double w = ftprl->get_w(z[feature_id], n[feature_id]);
        pred += value * w;
      }
      g0 = 2 * (pred - y[instance_id]);
      #pragma omp parallel for
      for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id); iter++) {
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