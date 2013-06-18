import numpy as np
import pdb

# Otrhogonalize Y against X, assumes column dimension of Y is less than X
# and that matrices are conformable.
def orthog(Y,X):
  dotY = np.dot(Y.transpose(),X).transpose()
  return (Y - np.dot(X,dotY))


def irlb(A,n,tol=0.0001,maxit=50):
  nu    = n
  m     = np.shape(A)[0]
  n     = np.shape(A)[1]
  m_b   = min((nu+4, 3*nu, n))  # Working dimension size
  mprod = 0
  it    = 0
  j     = 0
  k     = nu
  smax  = 1

# Check for minimum input dimensions and bail. XXX add me
  V  = np.zeros((n,m_b))
  W  = np.zeros((m,m_b))
  F  = np.zeros(n)
  B  = np.zeros((m_b,m_b))

  V[:,0]  = np.random.randn(n)   # Initial vector
  V[:,0]  = V[:,0]/np.linalg.norm(V)

  while(it < maxit):
    if(it>0): j=k
#    print("it=%d j=%d" % (it,j))
    W[:,j] = np.dot(A,V[:,j])     # W[,j] = A%*%V[,j]
    mprod+=1
    if(it>0):
      W[:,j] = orthog(W[:,j],W[:,0:j])
      # W[:,0:j] selects columns 0,1,...,j-1 apparently??
    s = np.linalg.norm(W[:,j])
    # XXX Add check for linear dependence...
    W[:,j] = W[:,j]/s
    # Lanczos process
    while(j<m_b):
      F = np.dot(A.transpose(),W[:,j])  # XXX Omit A.transpose()
      mprod+=1
      F = F - s*V[:,j]
      F = orthog(F,V[:,0:j+1])  # WTF is it with this indexing madness?
      fn = np.linalg.norm(F)
      F  = F/fn # XXX Add check for small fn
      if(j<m_b-1):   # XXX Right condition?
        V[:,j+1] = F
        B[j,j] = s
        B[j,j+1] = fn 
        W[:,j+1] = np.dot(A,V[:,j+1])
        mprod+=1
        # One step of classical Gram-Schmidt
        W[:,j+1] = W[:,j+1] - fn*W[:,j]
        # Full reorthogonalization
        W[:,j+1] = orthog(W[:,j+1],W[:,0:(j+1)])
        s = np.linalg.norm(W[:,j+1])  # XXX check for small s
        W[:,j+1] = W[:,j+1]/s
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
  return((U,S[1][0:nu],V))


######################################################
# Quick test:
######################################################
m  = 500
n  = 300
nu = 8
A = np.random.rand(m*n).reshape(m,n)

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
