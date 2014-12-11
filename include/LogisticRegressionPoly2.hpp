#pragma once

#include <memory>
#include "Matrix.hpp"
#include "FTPRL.hpp"

namespace FTPRL {

template<typename IndexType>
class LogisticRegressionPoly2 {
  
  bool is_manage_memory;
  
  double *z, *n;
  
  IndexType nfeature;
  
  FTPRL* ftprl;

public:
  
  LogisticRegressionPoly2(FTPRL* _ftprl, IndexType _nfeature) 
  : ftprl(_ftprl), z(new double[_nfeature]), n(new double[_nfeature]), nfeature(_nfeature), is_manage_memory(true)
  { }
  
  LogisticRegressionPoly2(FTPRL* _ftprl, IndexType _nfeature, double *_z, double *_n) 
  : ftprl(_ftprl), z(_z), n(_n), nfeature(_nfeature), is_manage_memory(false) 
  { } 
  
  virtual ~LogisticRegressionPoly2() {
    if (is_manage_memory) {
      delete [] z;
      delete [] n;
    }
  }
  
  template<typename ItorType, typename LabelType>
  void update(Matrix<IndexType, ItorType>* m, LabelType* y) {
    for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
      double pred = 0, g0 = 0;
      #pragma omp parallel for reduction(+:pred)
      for(ItorType iter1 = m->getFeatureItorBegin(instance_id);iter1 < m->getFeatureItorEnd(instance_id); iter1++) {
        IndexType feature_id1 = m->getFeatureId(iter1);
        double value1 = m->getValue(iter1);
        for(ItorType iter2 = m->getFeatureItorBegin(instance_id); iter2 <= iter1;iter2++) {
          IndexType feature_id2 = m->getFeatureId(iter2);
          double value2 = m->getValue(iter2);
          double w = ftprl->get_w(z[(feature_id1 ^ feature_id2) % nfeature], n[(feature_id1 ^ feature_id2) % nfeature]);
          pred += value1 * value2 * w;
        }
      }
      pred = sigma(pred);
      g0 = pred - y[instance_id];
      #pragma omp parallel for
      for(ItorType iter1 = m->getFeatureItorBegin(instance_id);iter1 < m->getFeatureItorEnd(instance_id); iter1++) {
        IndexType feature_id1 = m->getFeatureId(iter1);
        double value1 = m->getValue(iter1);
        for(ItorType iter2 = m->getFeatureItorBegin(instance_id);iter2 <= iter1; iter2++) {
          IndexType feature_id2 = m->getFeatureId(iter2);
          double value2 = m->getValue(iter2);
          double g = value1 * value2 * g0;
          ftprl->update_zn(g, z + (feature_id1 ^ feature_id2) % nfeature, n + (feature_id1 ^ feature_id2) % nfeature);
        }
      }
    }
  }

  template<typename ItorType, typename LabelType>
  void update(Matrix<IndexType, ItorType>* m, LabelType* y, LabelType skip_value) {
    for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
      if (y[instance_id] == skip_value) continue;
      double pred = 0, g0 = 0;
      #pragma omp parallel for reduction(+:pred)
      for(ItorType iter1 = m->getFeatureItorBegin(instance_id);iter1 < m->getFeatureItorEnd(instance_id); iter1++) {
        IndexType feature_id1 = m->getFeatureId(iter1);
        double value1 = m->getValue(iter1);
        for(ItorType iter2 = m->getFeatureItorBegin(instance_id);iter2 <= iter1; iter2++) {
          IndexType feature_id2 = m->getFeatureId(iter2);
          double value2 = m->getValue(iter2);
          double w = ftprl->get_w(z[(feature_id1 ^ feature_id2) % nfeature], n[(feature_id1 ^ feature_id2) % nfeature]);
          pred += value1 * value2 * w;
        }
      }
      pred = sigma(pred);
      g0 = pred - y[instance_id];
      #pragma omp parallel for
      for(ItorType iter1 = m->getFeatureItorBegin(instance_id);iter1 < m->getFeatureItorEnd(instance_id); iter1++) {
        IndexType feature_id1 = m->getFeatureId(iter1);
        double value1 = m->getValue(iter1);
        for(ItorType iter2 = m->getFeatureItorBegin(instance_id);iter2 <= iter1; iter2++) {
          IndexType feature_id2 = m->getFeatureId(iter2);
          double value2 = m->getValue(iter2);
          double g = value1 * value2 * g0;
          ftprl->update_zn(g, z + (feature_id1 ^ feature_id2) % nfeature, n + (feature_id1 ^ feature_id2) % nfeature);
        }
      }
    }
  }

  template<typename ItorType>
  void predict(Matrix<IndexType, ItorType>* m, double* y) {
    #pragma omp parallel for
    for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
      double& pred(y[instance_id]);
      pred = 0;
      for(ItorType iter1 = m->getFeatureItorBegin(instance_id);iter1 < m->getFeatureItorEnd(instance_id); iter1++) {
        IndexType feature_id1 = m->getFeatureId(iter1);
        double value1 = m->getValue(iter1);
        for(ItorType iter2 = m->getFeatureItorBegin(instance_id);iter2 <= iter1; iter2++) {
          IndexType feature_id2 = m->getFeatureId(iter2);
          double value2 = m->getValue(iter2);
          double w = ftprl->get_w(z[(feature_id1 ^ feature_id2) % nfeature], n[(feature_id1 ^ feature_id2) % nfeature]);
          pred += value1 * value2 * w;
        }
      }
      pred = sigma(pred);
    }
  }

  inline static double sigma(double x) {
    return 1 / (1 + std::exp(-x));
  }
  
};
  
}