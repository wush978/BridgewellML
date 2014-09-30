# distutils: language = c++

cimport ftprl
cimport matrix

cdef extern from "LogisticRegression.hpp" namespace "FTPRL":
    cdef cppclass LogisticRegression[IndexType]:
        LogisticRegression(ftprl.FTPRL*, IndexType)
        void update[ItorType, LabelType](matrix.Matrix[IndexType, ItorType]* m, LabelType* y)

cdef class PyLogisticRegression:
    cdef LogisticRegression[matrix.cINT32]* thisptr
    def __cinit__(self, pyftprl, nfeature):
        cdef matrix.cINT32 _nfeature = nfeature
        cdef ftprl.FTPRL* p = (<ftprl.PyFTPRL?>pyftprl).thisptr
        self.thisptr = new LogisticRegression[matrix.cINT32](p, _nfeature)
    def __dealloc__(self):
        del self.thisptr
    def update(self, matrix.PyScipySparseCSRMatrixProxy py_csr_matrix not None,  matrix.np.ndarray[matrix.cINT32, ndim = 1] py_y not None):
        self.thisptr.update[long, matrix.cINT32](py_csr_matrix.thisptr, &py_y[0])