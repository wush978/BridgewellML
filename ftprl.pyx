# distutils: language = c++

cdef extern from "<memory>" namespace "std":
    cdef cppclass shared_ptr[T]:
        shared_ptr(T*)

cdef extern from "FTPRL.hpp" namespace "FTPRL":
    cdef cppclass FTPRL:
        FTPRL(double, double, double, double)
        double alpha, beta, lambda1, lambda2;
        double get_w(double, double)
        void update_zn(double*, double*)

from cython.operator import dereference

cdef class PyFTPRL:
    cdef shared_ptr[FTPRL]* thisptr
    def __cinit__(self, double alpha, double beta, double lambda1, double lambda2):
        self.thisptr = new shared_ptr[FTPRL](new FTPRL(alpha, beta, lambda1, lambda2))
    def __dealloc__(self):
        del self.thisptr