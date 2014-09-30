cdef extern from "Matrix.hpp" namespace "FTPRL:":
    cdef cppclass Matrix[IndexType, ItorType]:
        Matrix()
        Matrix(IndexType, ItorType)
        IndexType getNFeature()
        IndexType getNInstance()

cdef extern from "ScipySparseCSRMatrixProxy.hpp":
    cdef cppclass ScipySparseCSRMatrixProxy[DataType, IndexType, ItorType](Matrix[IndexType, ItorType]):
        ScipySparseCSRMatrixProxy(IndexType, IndexType, IndexType*, IndexType*, DataType*)

cimport numpy as np
ctypedef np.int32_t cINT32
ctypedef np.double_t cDOUBLE

cdef class PyScipySparseCSRMatrixProxy:
    cdef ScipySparseCSRMatrixProxy[cDOUBLE, cINT32, long] *thisptr
