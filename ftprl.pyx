# distutils: language = c++

cimport ftprl

cdef class PyFTPRL:
    def __cinit__(self, double alpha, double beta, double lambda1, double lambda2):
        self.thisptr = new FTPRL(alpha, beta, lambda1, lambda2)
    def __dealloc__(self):
        del self.thisptr