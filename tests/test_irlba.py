######################################################
# Quick test:
######################################################

import numpy as np

def main():
    m = 10
    n = 10
    nu = 8
    A = np.random.rand(m*n).reshape(m, n)

    # Testing...Introduce near linear-dependece in the columns of A:
    A[:, 0] = A[:, 1] + 4*np.finfo(np.float).eps

    X = irlb(A, nu, tol=0.00001)

    # A gut check measurement of absolute decomposition error:
    print("TSVD: ||AV - US||_F = %f" \
          % np.linalg.norm(np.dot(A, X[2]) - X[0]*X[1], "fro"))

    # Compare estimated values with np.linalg.svd:
    S = np.linalg.svd(A, 0)

    abserr = np.max(np.abs(X[1]-S[1][0:nu]))
    print("Estmated/accurate singular values:")
    print(X[1])
    print(S[1][0:nu])
    print("||S_tsvd - S_svd||_inf = %f" % abserr)

