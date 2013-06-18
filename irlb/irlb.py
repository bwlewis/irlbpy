# irlb: Truncated SVD by implicitly restarted Lanczos bidiagonalization.
# Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
# J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005
#
# Implemented for Python by Bryan Lewis, Mike Kane, and Jim Baglama.
# Copyright (C) 2013 by B. W. Lewis, and Michael Kane.
# Contact: Bryan W. Lewis at blewis -at- illposed.net
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import pdb
import warnings

def orthog(Y,X):
  """Orthogonalize a vector or matrix Y against the columns of the matrix X.
  This function requires that the column dimension of Y is less than X and
  that Y and X have the same number of rows.
  """
  dotY = np.dot(Y.transpose(),X).transpose()
  return (Y - np.dot(X,dotY))

# Simple utility function used to check linear dependencies during computation:
def invcheck(x):
  eps2  = 2*np.finfo(np.float).eps
  if(x>eps2):
    x = 1/x
  else:
    x = 0
    warnings.warn("Ill-conditioning encountered, result accuracy may be poor")
  return(x)

def irlb(A,n,tol=0.0001,maxit=50):
  """Estimate a few of the largest singular values and corresponding singular
  vectors of matrix using the implicitly restarted Lanczos bidiagonalization
  method of Baglama and Reichel, see:

  Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
  J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005

  Keyword arguments:
  tol   -- An estimation tolerance. Smaller means more accurate estimates.
  maxit -- Maximum number of Lanczos iterations allowed.

  Given an input matrix A of dimension j * k, and an input desired number
  of singular values n, the function returns a tuple X with five entries:

  X[0] A j * nu matrix of estimated left singular vectors.
  X[1] A vector of length nu of estimated singular values.
  X[2] A k * nu matrix of estimated right singular vectors.
  X[3] The number of Lanczos iterations run.
  X[4] The number of matrix-vector products run.

  The algorithm estimates the truncated singular value decomposition:
  np.dot(A, X[2]) = X[0]*X[1].
  """
  nu    = n
  m     = np.shape(A)[0]
  n     = np.shape(A)[1]
  if(min(m,n)<2):
    raise Exception("The input matrix must be at least 2x2.")
  m_b   = min((nu+4, 3*nu, n))  # Working dimension size
  mprod = 0
  it    = 0
  j     = 0
  k     = nu
  smax  = 1

  V  = np.zeros((n,m_b))
  W  = np.zeros((m,m_b))
  F  = np.zeros(n)
  B  = np.zeros((m_b,m_b))

  V[:,0]  = np.random.randn(n) # Initial vector
  V[:,0]  = V[:,0]/np.linalg.norm(V)

  while(it < maxit):
    if(it>0): j=k
    W[:,j] = np.dot(A,V[:,j]) # W[,j] = A%*%V[,j]
    mprod+=1
    if(it>0):
      W[:,j] = orthog(W[:,j],W[:,0:j])
      # W[:,0:j] selects columns 0,1,...,j-1 apparently?? Arrgh.
    s = np.linalg.norm(W[:,j])
    sinv = invcheck(s)
    W[:,j] = sinv*W[:,j]
    # Lanczos process
    while(j<m_b):
      F = np.transpose(np.dot(W[:,j].transpose(),A))
      mprod+=1
      F = F - s*V[:,j]
      F = orthog(F,V[:,0:j+1])  # WTF is it with this indexing madness?
      fn = np.linalg.norm(F)
      fninv= invcheck(fn)
      F  = fninv * F
      if(j<m_b-1):
        V[:,j+1] = F
        B[j,j] = s
        B[j,j+1] = fn 
        W[:,j+1] = np.dot(A,V[:,j+1])
        mprod+=1
        # One step of classical Gram-Schmidt
        W[:,j+1] = W[:,j+1] - fn*W[:,j]
        # Full reorthogonalization
        W[:,j+1] = orthog(W[:,j+1],W[:,0:(j+1)])
        s = np.linalg.norm(W[:,j+1])
        sinv = invcheck(s) 
        W[:,j+1] = sinv * W[:,j+1]
      else:
        B[j,j] = s
      j+=1
    # End of Lanczos process
    S    = np.linalg.svd(B)
    R    = fn * S[0][m_b-1,:] # Residuals
    if(iter<1):
      smax = S[1][0]  # Largest Ritz value
    else:
      smax = max((S[1][0],smax))

    conv = sum(np.abs(R[0:nu]) < tol*smax)
#    print("iter=%d conv=%d" % (it,conv))
#    pdb.set_trace()  # browser--uncomment to drop to debug shell
    if(conv < nu):  # Not coverged yet XXX
      k = max(conv+nu,k)
      k = min(k,m_b-3)
    else:
      break
    # Use Ritz vectors
    V[:,0:k] = np.dot(V[:,0:m_b],S[2].transpose()[:,0:k])
    V[:,k] = F 
    B = np.zeros((m_b,m_b))
    # This sucks, must be better way to assign diagonal.
    for l in xrange(0,k):
      B[l,l] = S[1][l]
    B[0:k,k] = R[0:k]
    # Update the left approximate singular vectors
    W[:,0:k] = np.dot(W[:,0:m_b], S[0][:,0:k])
    it+=1

  U = np.dot(W[:,0:m_b], S[0][:,0:nu])
  V = np.dot(V[:,0:m_b], S[2].transpose()[:,0:nu])
  return((U,S[1][0:nu],V,it,mprod))


######################################################
# Quick test:
######################################################
m  = 10
n  = 10
nu = 8
A = np.random.rand(m*n).reshape(m,n)

# Testing...Introduce near linear-dependece in the columns of A:
A[:,0] = A[:,1] + 4*np.finfo(np.float).eps

X = irlb(A,nu,tol=0.00001)

# A gut check measurement of absolute decomposition error:
print("TSVD: ||AV - US||_F = %f" % np.linalg.norm(np.dot(A,X[2]) - X[0]*X[1],"fro"))

# Compare estimated values with np.linalg.svd:
S = np.linalg.svd(A, 0)

abserr = np.max(np.abs(X[1]-S[1][0:nu]))
print("Estmated/accurate singular values:")
print(X[1])
print(S[1][0:nu])
print("||S_tsvd - S_svd||_inf = %f" % abserr)
