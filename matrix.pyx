# distutils: language = c++

cdef extern from "ScipySparseCSRMatrixProxy.hpp":
    cdef cppclass ScipySparseCSRMatrixProxy[DataType, IndexType, ItorType]:
        ScipySparseCSRMatrixProxy(IndexType, IndexType, IndexType*, IndexType*, DataType*)

import cython

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

ctypedef np.int32_t cINT32
ctypedef np.double_t cDOUBLE

cdef class PyScipySparseCSRMatrixProxy:
    cdef ScipySparseCSRMatrixProxy[cDOUBLE, cINT32, long] *thisptr
    def __cinit__(self, m):
        cdef cINT32 nfeature, ninstance
        ninstance = m.shape[0]
        nfeature = m.shape[1]
        cdef np.ndarray[cINT32, ndim = 1] indices, indptr
        indices = m.indices
        indptr = m.indptr
        cdef np.ndarray[cDOUBLE, ndim = 1] data
        data = m.data
        self.thisptr = new ScipySparseCSRMatrixProxy[cDOUBLE, cINT32, long](nfeature, ninstance, &indices[0], &indptr[0], &data[0])
    def __dealloc__(self):
        del self.thisptr
