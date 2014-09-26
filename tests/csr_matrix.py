from scipy.sparse import csr_matrix
from scipy import array
indptr = array([0,2,3,6])
indices = array([0,2,2,0,1,2])
data = array([0.1,0.2,3,4,5,6])
a = csr_matrix( (data,indices,indptr), shape=(3,3) )
print a
import matrix
print matrix.PyScipySparseCSRMatrixProxy(a)
