irlbpy
======

Truncated SVD by implicitly restarted Lanczos bidiagonalization for Numpy!

irlb: A fast and memory-efficient method for estimating a few largest signular values and corresponding singular vectors of very large matrices.

Adapted from the algorithm by Jim Baglama and Lothar Reichel:
Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005.

Installation:
---

There are several options for installing the irlbpy package. The easiest is 
to simply ``pip install`` the code (either into your system site-packages or
a `virtualenv <https://pypi.python.org/pypi/virtualenv>` with the command::

```
pip install -e git+https://github.com/bwlewis/irlbpy.git#egg=irlb
```

Otherwise, if you have downloaded the code you can install the package 
locally by executing the following commands from the project's home directory:

```
python setup.py sdist
pip install dist/irlbpy-0.1.0.tar.gz 
```

Usage:
---

```
S = irlb(A, n, [tol=0.0001 [, maxit=50]])
```
Where, A is a double-precision-valued matrix, n is the number of singular values and corresponding singular values to compute, tol is an optional convergence tolerance parameter that controls the accuracy of the estimated singular values, and maxit is an optional limit on the maximum number of Lanczos iterations.

The returned triple S contains the matrix of left singular vectors, a vector of singular values, and the matrix of right singular vectors, respectively, such that:
```
np.dot(A, S[2]) - S[0]*S[1]
```
is small.
