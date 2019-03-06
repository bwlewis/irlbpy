######################################################
# Quick test:
######################################################

import numpy as np
import scipy.sparse as sp
from irlb import *

def main():
    m = 2000   # Number of rows
    n = 2000   # Number of columns
    nnz = 10000  # Maximum number of nonzero elements
    nu = 5      # Number of singular values to compute

    v = np.random.rand(nnz)
    i = np.random.randint(0, high=m, size=nnz)
    j = np.random.randint(0, high=n, size=nnz)
    A = sp.csc_matrix((v, (i, j)), shape=(m, n))
    X = irlb(A, nu, tol=0.000001)
    # A gut check measurement of absolute decomposition error:
    print("TSVD: ||AV - US||_F = %f" \
          % np.linalg.norm(A.dot(sp.csr_matrix(X[2])).todense().A - X[0]*X[1],"fro"))

    # Compare with reference svd
    S = np.linalg.svd(A.todense(), 0)
    abserr = np.max(np.abs(X[1]-S[1][0:nu]))
    print("Estmated/accurate singular values:")
    print(X[1])
    print(S[1][0:nu])
    print("||S_tsvd - S_svd||_inf = %f" % abserr)
