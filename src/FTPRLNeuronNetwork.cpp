#include <memory>
#include <Rcpp.h>
#include "NumericMatrixProxy.hpp"
#include "dgCMatrixProxy.hpp"
#include "FTPRLProxy.hpp"
#include "NeuronNetwork.hpp"

using namespace Rcpp;

template<typename InputType, typename MatrixProxy, typename IndexType>
void update_FTPRLNeuronNetwork(InputType Rm, LogicalVector y, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> plearner(new FTPRLProxy(Rlearner));
  
  IntegerVector nnode(Rlearner.slot("nnode"));
  List zlist(Rlearner.slot("z")), nlist(Rlearner.slot("n"));
  std::vector<double*> z(nnode.size() - 1, NULL), n(nnode.size() - 1, NULL);
  for(int i = 0;i + 1 < nnode.size();i++) {
    z[i] = REAL(zlist[i]);
    n[i] = REAL(nlist[i]);
  }
  IndexType *pnnode = &nnode[0];
  FTPRL::NeuronNetwork<IndexType> lr(plearner.get(), nnode.size(), pnnode, &z[0], &n[0]);
  lr.update(&m, &y[0]);
}

//'@export
// [[Rcpp::export("update_FTPRLNeuronNetwork.matrix")]]
void update_FTPRLNeuronNetwork_matrix(NumericMatrix Rm, LogicalVector y, S4 Rlearner) {
  if (Rm.nrow() != y.size()) throw std::invalid_argument("");
  update_FTPRLNeuronNetwork<NumericMatrix, NumericMatrixProxy, int>(Rm, y, Rlearner);
}

//'@export
// [[Rcpp::export("update_FTPRLNeuronNetwork.dgCMatrix")]]
void update_FTPRLNeuronNetwork_dgCMatrix(S4 Rm, LogicalVector y, S4 Rlearner) {
  IntegerVector dim(Rm.slot("Dim"));
  if (dim[1] != y.size()) throw std::invalid_argument("");
  update_FTPRLNeuronNetwork<S4, dgCMatrixProxy, int>(Rm, y, Rlearner);
}

/**
template<typename InputType, typename MatrixProxy, typename IndexType>
SEXP predict_FTPRLNeuronNetwork(InputType Rm, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> learner(new FTPRLProxy(Rlearner));
  double *z = REAL(Rlearner.slot("z")), *n = REAL(Rlearner.slot("n"));
  NumericVector retval(m.getNInstance(), 0.0);
  double *pretval = REAL(wrap(retval));
  FTPRL::NeuronNetwork<IndexType> lr(learner.get(), m.getNFeature(), z, n);
  lr.predict(&m, pretval);
  return retval;
}

//'@export
// [[Rcpp::export("predict_FTPRLNeuronNetwork.matrix")]]
SEXP predict_FTPRLNeuronNetwork_matrix(NumericMatrix Rm, S4 Rlearner) {
  return predict_FTPRLNeuronNetwork<NumericMatrix, NumericMatrixProxy, int>(Rm, Rlearner);
}

//'@export
// [[Rcpp::export("predict_FTPRLNeuronNetwork.dgCMatrix")]]
SEXP predict_FTPRLNeuronNetwork_dgCMatrix(S4 Rm, S4 Rlearner) {
  return predict_FTPRLNeuronNetwork<S4, dgCMatrixProxy, int>(Rm, Rlearner);
}
**/