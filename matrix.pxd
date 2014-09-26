cdef extern from "ScipySparseCSRMatrixProxy.hpp":
    cdef cppclass ScipySparseCSRMatrixProxy[DataType, IndexType, ItorType]:
        ScipySparseCSRMatrixProxy(IndexType, IndexType, IndexType*, IndexType*, DataType*)

cimport numpy as np
ctypedef np.int32_t cINT32
ctypedef np.double_t cDOUBLE

