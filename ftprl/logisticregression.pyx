# distutils: language = c++

cimport ftprl
cimport matrix

cdef extern from "LogisticRegression.hpp" namespace "FTPRL":
    cdef cppclass LogisticRegression[IndexType]:
        LogisticRegression(ftprl.FTPRL*, IndexType)
        void update[ItorType, LabelType](matrix.Matrix[IndexType, ItorType]* m, LabelType* y)
        void predict[ItorType](matrix.Matrix[IndexType, ItorType]* m, double* y)

cdef class PyLogisticRegression:
    cdef LogisticRegression[matrix.cINT32]* thisptr
    def __cinit__(self, ftprl.PyFTPRL pyftprl not None, matrix.cINT32 nfeature):
        self.thisptr = new LogisticRegression[matrix.cINT32](pyftprl.thisptr, nfeature)
    def __dealloc__(self):
        del self.thisptr
    def update(self, matrix.PyScipySparseCSRMatrixProxy py_csr_matrix not None,  matrix.np.ndarray[matrix.cINT32, ndim = 1] py_y not None):
        self.thisptr.update[long, matrix.cINT32](py_csr_matrix.thisptr, &py_y[0])
    def predict(self, matrix.PyScipySparseCSRMatrixProxy py_csr_matrix not None, matrix.np.ndarray[matrix.cDOUBLE, ndim = 1] py_y not None):
        self.thisptr.predict[long](py_csr_matrix.thisptr, &py_y[0])