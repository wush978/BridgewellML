cdef extern from "FTPRL.hpp" namespace "FTPRL":
    cdef cppclass FTPRL:
        FTPRL(double, double, double, double)
        double alpha, beta, lambda1, lambda2;
        double get_w(double, double)
        void update_zn(double*, double*)

cdef class PyFTPRL:
    cdef FTPRL* thisptr
