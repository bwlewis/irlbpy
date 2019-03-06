import numpy as np
import scipy.sparse as sp
import warnings


__all__ = ["irlb"]

def mult(A, x, t=False):
    """Matrix-vector product wrapper
    A is a numpy 2d array or matrix, or a scipy matrix or sparse matrix.
    x is a numpy vector only.
    Compute A.dot(x) if t is False,
            A.transpose().dot(x)  otherwise.
    """
    if sp.issparse(A):
        m = A.shape[0]
        n = A.shape[1]
        if t:
            return sp.csr_matrix(x).dot(A).transpose().todense().A[:, 0]
        return A.dot(sp.csr_matrix(x).transpose()).todense().A[:, 0]
    if t:
        return x.dot(A)
    return A.dot(x)


def orthog(Y, X):
    """Orthogonalize a vector or matrix Y against the columns of the matrix X.
    This function requires that the column dimension of Y is less than X and
    that Y and X have the same number of rows.
    """
    dotY = mult(X, Y, t=True)
    return Y - mult(X, dotY)


def invcheck(x):
    """Simple utility function used to check linear dependencies during computation
    """
    eps2 = 2 * np.finfo(np.float).eps
    if x > eps2:
        x = 1 / x
    else:
        x = 0
        mes = "Ill-conditioning encountered, result accuracy may be poor"
        warnings.warn(mes)
    return x


def irlb(A, n, tol=0.0001, maxit=50):
    """Estimate a few of the largest singular values and corresponding singular
    vectors of matrix using the implicitly restarted Lanczos bidiagonalization
    method of Baglama and Reichel, see
    Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
    J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005

    Keyword arguments:
    tol   -- An estimation tolerance. Smaller means more accurate estimates.
    maxit -- Maximum number of Lanczos iterations allowed.
    
    Given an input matrix A of dimension j * k, and an input desired number
    of singular values n, the function returns a tuple X with five entries:

    Returns
    -------
    irlb : tuple
        Tuple with 5 items
            0. A j * nu matrix of estimated left singular vectors.
            1. A vector of length nu of estimated singular values.
            2. A k * nu matrix of estimated right singular vectors.
            3. The number of Lanczos iterations run.
            4. The number of matrix-vector products run.

    
    X[0] A j * nu matrix of estimated left singular vectors.
    X[1] A vector of length nu of estimated singular values.
    X[2] A k * nu matrix of estimated right singular vectors.
    X[3] The number of Lanczos iterations run.
    X[4] The number of matrix-vector products run.

    The algorithm estimates the truncated singular value decomposition:
    A.dot(X[2]) = X[0]*X[1].
    """
    nu = n
    m = A.shape[0]
    n = A.shape[1]
    if min(m, n) < 2:
        raise Exception("The input matrix must be at least 2x2.")
    m_b = min((nu + 20, 3 * nu, n))  # Working dimension size
    mprod = 0
    it = 0
    j = 0
    k = nu
    smax = 1
    sparse = sp.issparse(A)

    V = np.zeros((n, m_b))
    W = np.zeros((m, m_b))
    F = np.zeros((n, 1))
    B = np.zeros((m_b, m_b))

    V[:, 0] = np.random.randn(n)  # Initial vector
    V[:, 0] = V[:, 0] / np.linalg.norm(V)

    while it < maxit:
        if it > 0:
            j = k
        W[:, j] = mult(A, V[:, j])
        mprod += 1
        if it > 0:
            W[:, j] = orthog(W[:, j], W[:, 0:j])
        s = np.linalg.norm(W[:, j])
        sinv = invcheck(s)
        W[:, j] = sinv * W[:, j]
        # Lanczos process
        while j < m_b:
            F = mult(A, W[:, j], t=True)
            mprod += 1
            F = F - s*V[:, j]
            F = orthog(F, V[:, 0:j+1])
            fn = np.linalg.norm(F)
            fninv = invcheck(fn)
            F = fninv * F
            if j < (m_b - 1):
                V[:, j+1] = F
                B[j, j] = s
                B[j, j+1] = fn
                W[:, j+1] = mult(A, V[:, j+1])
                mprod += 1
                # One step of classical Gram-Schmidt...
                W[:, j+1] = W[:, j+1] - fn*W[:, j]
                # ...with full reorthogonalization
                W[:, j+1] = orthog(W[:, j+1], W[:, 0:j+1])
                s = np.linalg.norm(W[:, j+1])
                sinv = invcheck(s)
                W[:, j+1] = sinv * W[:, j+1]
            else:
                B[j, j] = s
            j += 1
        # End of Lanczos process
        S = np.linalg.svd(B)
        R = fn * S[0][m_b-1, :]  # Residuals
        if it < 1:
            smax = S[1][0]  # Largest Ritz value
        else:
            smax = max((S[1][0], smax))

        conv = sum(np.abs(R[0:nu]) < tol*smax)
        if conv < nu:  # Not coverged yet
            k = max(conv+nu, k)
            k = min(k, m_b-3)
        else:
            break
        # Update the Ritz vectors
        V[:, 0:k] = V[:, 0:m_b].dot(S[2].transpose()[:, 0:k])
        V[:, k] = F
        B = np.zeros((m_b, m_b))
        
        # Improve this! There must be better way to assign diagonal...
        #for l in range(0, k):
        #    B[l, l] = S[1][l]
        np.fill_diagonal(B, S[1])
        
        B[0:k, k] = R[0:k]
        # Update the left approximate singular vectors
        W[:, 0:k] = W[:, 0:m_b].dot(S[0][:, 0:k])
        it += 1

    U = W[:, 0:m_b].dot(S[0][:, 0:nu])
    V = V[:, 0:m_b].dot(S[2].transpose()[:, 0:nu])
    return (U, S[1][0:nu], V, it, mprod)
