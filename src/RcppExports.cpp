// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// update_FTPRLLogisticRegression_matrix
void update_FTPRLLogisticRegression_matrix(NumericMatrix Rm, LogicalVector y, S4 Rlearner);
RcppExport SEXP BridgewellML_update_FTPRLLogisticRegression_matrix(SEXP RmSEXP, SEXP ySEXP, SEXP RlearnerSEXP) {
BEGIN_RCPP
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< NumericMatrix >::type Rm(RmSEXP );
        Rcpp::traits::input_parameter< LogicalVector >::type y(ySEXP );
        Rcpp::traits::input_parameter< S4 >::type Rlearner(RlearnerSEXP );
        update_FTPRLLogisticRegression_matrix(Rm, y, Rlearner);
    }
    return R_NilValue;
END_RCPP
}
// update_FTPRLLogisticRegression_dgCMatrix
void update_FTPRLLogisticRegression_dgCMatrix(S4 Rm, LogicalVector y, S4 Rlearner);
RcppExport SEXP BridgewellML_update_FTPRLLogisticRegression_dgCMatrix(SEXP RmSEXP, SEXP ySEXP, SEXP RlearnerSEXP) {
BEGIN_RCPP
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< S4 >::type Rm(RmSEXP );
        Rcpp::traits::input_parameter< LogicalVector >::type y(ySEXP );
        Rcpp::traits::input_parameter< S4 >::type Rlearner(RlearnerSEXP );
        update_FTPRLLogisticRegression_dgCMatrix(Rm, y, Rlearner);
    }
    return R_NilValue;
END_RCPP
}
// predict_FTPRLLogisticRegression_matrix
SEXP predict_FTPRLLogisticRegression_matrix(NumericMatrix Rm, S4 Rlearner);
RcppExport SEXP BridgewellML_predict_FTPRLLogisticRegression_matrix(SEXP RmSEXP, SEXP RlearnerSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< NumericMatrix >::type Rm(RmSEXP );
        Rcpp::traits::input_parameter< S4 >::type Rlearner(RlearnerSEXP );
        SEXP __result = predict_FTPRLLogisticRegression_matrix(Rm, Rlearner);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// predict_FTPRLLogisticRegression_dgCMatrix
SEXP predict_FTPRLLogisticRegression_dgCMatrix(S4 Rm, S4 Rlearner);
RcppExport SEXP BridgewellML_predict_FTPRLLogisticRegression_dgCMatrix(SEXP RmSEXP, SEXP RlearnerSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< S4 >::type Rm(RmSEXP );
        Rcpp::traits::input_parameter< S4 >::type Rlearner(RlearnerSEXP );
        SEXP __result = predict_FTPRLLogisticRegression_dgCMatrix(Rm, Rlearner);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
// update_FTPRLNeuronNetwork_matrix
void update_FTPRLNeuronNetwork_matrix(NumericMatrix Rm, LogicalVector y, S4 Rlearner);
RcppExport SEXP BridgewellML_update_FTPRLNeuronNetwork_matrix(SEXP RmSEXP, SEXP ySEXP, SEXP RlearnerSEXP) {
BEGIN_RCPP
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< NumericMatrix >::type Rm(RmSEXP );
        Rcpp::traits::input_parameter< LogicalVector >::type y(ySEXP );
        Rcpp::traits::input_parameter< S4 >::type Rlearner(RlearnerSEXP );
        update_FTPRLNeuronNetwork_matrix(Rm, y, Rlearner);
    }
    return R_NilValue;
END_RCPP
}
// update_FTPRLNeuronNetwork_dgCMatrix
void update_FTPRLNeuronNetwork_dgCMatrix(S4 Rm, LogicalVector y, S4 Rlearner);
RcppExport SEXP BridgewellML_update_FTPRLNeuronNetwork_dgCMatrix(SEXP RmSEXP, SEXP ySEXP, SEXP RlearnerSEXP) {
BEGIN_RCPP
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< S4 >::type Rm(RmSEXP );
        Rcpp::traits::input_parameter< LogicalVector >::type y(ySEXP );
        Rcpp::traits::input_parameter< S4 >::type Rlearner(RlearnerSEXP );
        update_FTPRLNeuronNetwork_dgCMatrix(Rm, y, Rlearner);
    }
    return R_NilValue;
END_RCPP
}
