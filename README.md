irlbpy
======

Truncated SVD by implicitly restarted Lanczos bidiagonalization for Numpy!


irlb: A fast and memory-efficient method for estimating a few largest signular values and corresponding singular vectors of very large matrices.

Adapted from the algorithm by Jim Baglama and Lothar Reichel:
Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005.

Usage:
```
S = irlb(A, n, [tol=0.0001 [, maxit=50]])
```
Where, A is a double-precision-valued matrix, n is the number of singular values and corresponding singular values to compute, tol is an optional convergence tolerance parameter that controls the accuracy of the estimated singular values, and maxit is an optional limit on the maximum number of Lanczos iterations.

The returned triple S contains the matrix of left singular vectors, a vector of singular values, and the matrix of right singular vectors, respectively, such that:
```
np.dot(A, S[2]) - S[0]*S[1]
```
is small.
