#pragma once

#include <memory>
#include "Matrix.hpp"
#include "FTPRL.hpp"

namespace FTPRL {

template<typename IndexType>
class NeuronNetwork {

  FTPRL* ftprl;

  bool is_manage_memory;
 
  IndexType nlayer;

  IndexType* nnode;

  double **z, **n;

public:

  NeuronNetwork(FTPRL* _ftprl, IndexType _nlayer, IndexType* _nnode)
  : ftprl(_ftprl), nlayer(_nlayer), nnode(_nnode), is_manage_memory(true)
  {
    if (nlayer < 3) throw std::logic_error("Less than 3 layers");
    z = new double*[nlayer - 1];
    n = new double*[nlayer - 1];
    for(IndexType i = 0;i + 1 < nlayer;i++) {
      // i \in [0, nlayer - 2]
      IndexType current_nodes = nnode[i], next_nodes = nnode[i + 1];
      z[i] = new double[current_nodes * next_nodes];
      ::memset(z[i], 0, sizeof(double) * current_nodes * next_nodes);
      n[i] = new double[current_nodes * next_nodes];
      ::memset(z[i], 0, sizeof(double) * current_nodes * next_nodes);
    }
  }
  
  NeuronNetwork(FTPRL* _ftprl, IndexType _nlayer, IndexType* _nnode, double **_z, double **_n)
  : ftprl(_ftprl), nlayer(_nlayer), nnode(_nnode), z(_z), n(_n), is_manage_memory(false)
  {
    if (nlayer < 3) throw std::logic_error("Less than 3 layers");
  }

  virtual ~NeuronNetwork() {
    if (is_manage_memory) {
      for(IndexType i = 0;i + 1 < nlayer;i++) {
        delete [] z[i];
        delete [] n[i];
      }
      delete [] z;
      delete [] n;
    }
  }
  
  template<typename ItorType, typename LabelType>
  void update(Matrix<IndexType, ItorType>* m, LabelType* y) {
    if (m->getNFeature() != nnode[nlayer - 1]) throw std::invalid_argument("Inconsistent nfeature");
    typedef std::vector<double> NumVec;
    typedef std::vector<NumVec> NumVecVec;
    NumVecVec node_value(nlayer - 1, NumVec()), g0(nlayer - 1, NumVec());
    for(IndexType i = 0;i + 1 < nlayer;i++) {
      node_value[i].resize(nnode[i], 0.0);
      g0[i].resize(nnode[i], 0.0);
    }
    #pragma omp parallel
    for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
      for(IndexType l = nlayer - 1;l > 0;l--) { // l \in [1, nlayer - 1]
        #pragma omp for
        for(IndexType ll = 0;ll < nnode[l - 1];ll++) {
          // ll \in [0, nnode[l - 1]]
          node_value[l - 1][ll] = 0;
        }
        IndexType ln = nnode[l];
        if (l == nlayer - 1) {
          // last layer
          #pragma omp for
          for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id);iter++) {
            IndexType feature_id = m->getFeatureId(iter);
            double value = m->getValue(iter);
            for(IndexType ll = 0;ll < nnode[l - 1];ll++) {
              double w = ftprl->get_w(z[l - 1][ll * ln + feature_id], n[l - 1][ll * ln + feature_id]);
              // ll \in [0, nnode[l - 1]]
              node_value[l - 1][ll] += value * w;
            }
          }
        } else { // l < nlayer - 1
          #pragma omp for
          for(IndexType feature_id = 0;feature_id < nnode[l];feature_id++) {
            double value = node_value[l][feature_id];
            for(IndexType ll = 0;ll < nnode[l - 1];ll++) {
              double w = ftprl->get_w(z[l - 1][ll * ln + feature_id], n[l - 1][ll * ln + feature_id]);
              node_value[l - 1][ll] += value * w;
            }
          }
        } // l < nlayer - 1
        #pragma omp for
        for(IndexType ll = 0;ll < nnode[l - 1];ll++) {
          node_value[l - 1][ll] = sigma(node_value[l - 1][ll]);
        }
      } // l
      double yt = y[instance_id];
      for(IndexType l = 0;l + 1 < nlayer;l++) {
        IndexType ln = nnode[l];
        if (l == 0) {
          #pragma omp for
          for(IndexType ll = 0;ll < nnode[l];ll++) {
            g0[l][ll] = (1 - yt) / (1 - node_value[l][ll]) - yt / node_value[l][ll];
          }
        } else {
          #pragma omp for
          for(IndexType ll = 0;ll < nnode[l];ll++) {
            g0[l][ll] = 0;
            for(IndexType llp = 0;llp < nnode[l - 1];llp++) {
              double w = ftprl->get_w(z[l - 1][llp * ln + ll], n[l - 1][llp * ln + ll]);
              g0[l][ll] += node_value[l - 1][llp] * (1 - node_value[l - 1][llp]) * w * g0[l - 1][llp];
            }
          }
        }
        for(IndexType l = 0; l + 1 < nlayer;l++) {
          IndexType ln = nnode[l];
          #pragma omp for
          for(IndexType ll = 0;ll < nnode[l];ll++) {
            if (l + 2 == nlayer) {
              for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id);iter++) {
                IndexType feature_id = m->getFeatureId(iter);
                double value = m->getValue(iter);
                double g = g0[l][ll] * node_value[l][ll] * (1 - node_value[l][ll]) * value;
                ftprl->update_zn(g, z[l] + ll * ln + feature_id, n[l] + ll * ln + feature_id);
              }
            } else {
              for(IndexType feature_id = 0;feature_id < nnode[l + 1];feature_id++) {
                double value = node_value[l + 1][feature_id];
                double g = g0[l][ll] * node_value[l][ll] * (1 - node_value[l][ll]) * value;
                ftprl->update_zn(g, z[l] + ll * ln + feature_id, n[l] + ll * ln + feature_id);
              }
            } // l + 2 < nlayer
          }
        }
      }
    }
  }

  template<typename ItorType>
  void predict(Matrix<IndexType, ItorType>* m, double* y) {
    if (m->getNFeature() != nnode[nlayer - 1]) throw std::invalid_argument("Inconsistent nfeature");
    typedef std::vector<double> NumVec;
    typedef std::vector<NumVec> NumVecVec;
    #pragma omp parallel
    {
      NumVecVec node_value(nlayer - 1, NumVec());
      for(IndexType i = 0;i + 1 < nlayer;i++) {
        node_value[i].resize(nnode[i], 0.0);
      }
      #pragma omp for
      for(IndexType instance_id = 0;instance_id < m->getNInstance();instance_id++) {
        for(IndexType l = nlayer - 1;l > 0;l--) { // l \in [1, nlayer - 1]
          for(IndexType ll = 0;ll < nnode[l - 1];ll++) {
            // ll \in [0, nnode[l - 1]]
            node_value[l - 1][ll] = 0;
          }
          IndexType ln = nnode[l];
          if (l == nlayer - 1) {
            // last layer
            for(ItorType iter = m->getFeatureItorBegin(instance_id);iter < m->getFeatureItorEnd(instance_id);iter++) {
              IndexType feature_id = m->getFeatureId(iter);
              double value = m->getValue(iter);
              for(IndexType ll = 0;ll < nnode[l - 1];ll++) {
                double w = ftprl->get_w(z[l - 1][ll * ln + feature_id], n[l - 1][ll * ln + feature_id]);
                // ll \in [0, nnode[l - 1]]
                node_value[l - 1][ll] += value * w;
              }
            }
          } else { // l < nlayer - 1
            for(IndexType feature_id = 0;feature_id < nnode[l];feature_id++) {
              double value = node_value[l][feature_id];
              for(IndexType ll = 0;ll < nnode[l - 1];ll++) {
                double w = ftprl->get_w(z[l - 1][ll * ln + feature_id], n[l - 1][ll * ln + feature_id]);
                node_value[l - 1][ll] += value * w;
              }
            }
          } // l < nlayer - 1
          for(IndexType ll = 0;ll < nnode[l - 1];ll++) {
            node_value[l - 1][ll] = sigma(node_value[l - 1][ll]);
          }
        } // l
        y[instance_id] = node_value[0][0];
      }
    }
  }
};

}