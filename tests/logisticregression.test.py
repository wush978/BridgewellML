from scipy.sparse import csr_matrix
from scipy import array
indptr = array([0,2,3,6])
indices = array([0,2,2,0,1,2])
data = array([0.1,0.2,3,4,5,6])
a = csr_matrix( (data,indices,indptr), shape=(3,3) )
print a
import matrix
_a = matrix.PyScipySparseCSRMatrixProxy(a)
print _a.getNFeature() 
print _a.getNInstance()
import numpy as np
y = array([0, 1, 0], dtype = np.int32)

from ftprl import PyFTPRL
param = PyFTPRL(0.1, 1, 0.1, 0.0)

from logisticregression import PyLogisticRegression
learner = PyLogisticRegression(param, 3)
retval = array([0.0, 0.0, 0.0])
learner.update(_a, y)
learner.predict(_a, retval)
print retval
learner.update(_a, y)
learner.predict(_a, retval)
print retval
learner.update(_a, y)
learner.predict(_a, retval)
print retval
learner.update(_a, y)
learner.predict(_a, retval)
print retval
learner.update(_a, y)
learner.predict(_a, retval)
print retval
learner.update(_a, y)
learner.predict(_a, retval)
print retval
