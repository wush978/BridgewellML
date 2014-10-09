#include <memory>
#include <Rcpp.h>
#include "NumericMatrixProxy.hpp"
#include "dgCMatrixProxy.hpp"
#include "FTPRLProxy.hpp"
#include "NeuronNetwork.hpp"

using namespace Rcpp;

const std::vector<double*> getzn(S4 Rlearner, const std::string& name) {
  IntegerVector nnode(Rlearner.slot("nnode"));
  List list(Rlearner.slot(name));
  std::vector<double*> retval(nnode.size() - 1, NULL);
  for(int i = 0;i + 1 < nnode.size();i++) {
    retval[i] = REAL(list[i]);
  }
  return retval;
}

template<typename IndexType>
static FTPRL::NeuronNetwork<IndexType> init_learner(FTPRLProxy* p, int nlayer, int* nnode, std::vector<double*>& z, std::vector<double*>& n) {
  return FTPRL::NeuronNetwork<IndexType>(p, nlayer, nnode, &z[0], &n[0]);
}

template<typename InputType, typename MatrixProxy, typename IndexType>
void update_FTPRLNeuronNetwork(InputType Rm, LogicalVector y, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> plearner(new FTPRLProxy(Rlearner));
  std::vector<double*> z(getzn(Rlearner, "z")), n(getzn(Rlearner, "n"));
  IntegerVector nnode(Rlearner.slot("nnode"));
  int *pnode = &nnode[0];
  auto lr(init_learner<IndexType>(plearner.get(), nnode.size(), pnode, z, n));
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

template<typename InputType, typename MatrixProxy, typename IndexType>
SEXP predict_FTPRLNeuronNetwork(InputType Rm, S4 Rlearner) {
  MatrixProxy m(Rm);
  std::shared_ptr<FTPRLProxy> plearner(new FTPRLProxy(Rlearner));
  std::vector<double*> z(getzn(Rlearner, "z")), n(getzn(Rlearner, "n"));
  IntegerVector nnode(Rlearner.slot("nnode"));
  int *pnode = &nnode[0];
  auto lr(init_learner<IndexType>(plearner.get(), nnode.size(), pnode, z, n));
  NumericVector retval(m.getNInstance(), 0.0);
  double *pretval = REAL(wrap(retval));
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
