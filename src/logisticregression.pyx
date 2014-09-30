# distutils: language = c++

cimport ftprl
cimport matrix

cdef extern from "LogisticRegression.hpp" namespace "FTPRL":
    cdef cppclass LogisticRegression[IndexType]:
        LogisticRegression(ftprl.FTPRL*, IndexType)

cdef class PyLogisticRegression:
    cdef LogisticRegression[matrix.cINT32]* thisptr
    def __cinit__(self, pyftprl, nfeature):
        cdef matrix.cINT32 _nfeature = nfeature
        cdef ftprl.FTPRL* p = (<ftprl.PyFTPRL?>pyftprl).thisptr
        self.thisptr = new LogisticRegression[matrix.cINT32](p, _nfeature)
    def __dealloc__(self):
        del self.thisptr