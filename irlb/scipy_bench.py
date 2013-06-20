from __future__ import division, print_function
from irlb import *
from argparse import *
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

import time

def parse_args():
  parser = ArgumentParser()
  parser.add_argument("-m", "--num-row", type=int,
    help="The number of rows in the test matrix")
  parser.add_argument("-n", "--num-col", type=int,
    help="The number of columns in the test matrix")
  parser.add_argument("-v", "--nu", type=int,
    help="The number of desired singular vectors")
  parser.add_argument("-t", "--tol", default=0.0001,
    help="An estimation tolerance. Smaller means more accurate estimates."+
    " Defaults to 0.0001.")
  parser.add_argument("-i", "--max-it", default=50,
    help="The maximum number of Lanczos iterations allowed.")
  parser.add_argument("-c", "--csv", action="store_const", const=True,
    default=False,
    help="Should the output be csv formatted? If True then the return will "+
      "be: <IRB timing>,<SVD timing>,<absolue error>")
  parser.add_argument("-s", "--sparsity", default=0., type=float,
    help="The approximate sparsity of the matrix. Default is 0 (dense).")
  parser.add_argument("-l", "--irlb-only", action="store_const", const=True,
    default=False)
  parser.add_argument("-p", "--scipy-only", action="store_const", const=True,
    default=False)

  return parser.parse_args()

def main():
  args = parse_args()
  if not args.scipy_only:
    if args.sparsity > 0.:
      if args.sparsity >= 1.:
        raise(RuntimeError("Sparsity must be less than 1."))
      nnz = (1.-args.sparsity) * args.num_row * args.num_col
      v = np.random.rand(nnz)
      i = np.random.randint(0, high=args.num_row, size=nnz)
      j = np.random.randint(0, high=args.num_col, size=nnz)
      A = sp.csc_matrix((v,(i,j)), shape=(args.num_row, args.num_col))
    else:
      A=np.random.rand(args.num_row*args.num_col).reshape(
        args.num_row,args.num_col)

    start_irlb = time.time()
    X = irlb(A, args.nu, tol=args.tol, maxit=args.max_it)
    end_irlb = time.time()

  # Compare estimated values with np.linalg.svd:
  if not args.irlb_only:
    if args.sparsity > 0.:
      start_svd = time.time()
      S = splinalg.svds(A=A, k=args.nu, tol=args.tol, maxiter=args.max_it,
        return_singular_vectors=True)
      end_svd = time.time()
    else:
      start_svd = time.time()
      S = np.linalg.svd(A, 0)
      end_svd = time.time()

    abserr = np.max(np.abs(X[1]-S[1][0:args.nu]))

  if (args.csv):
    if args.irlb_only:
      print("{}".format(end_irlb-start_irlb))
    elif args.scipy_only:
      print("{}".format(end_svd-start_svd))
    else:
      print("{},{},{}".format(end_irlb-start_irlb, end_svd-start_svd, abserr))
  else:
    if not args.scipy_only:
      print("IRLB timing: {}".format(end_irlb-start_irlb))
    if not args.irlb_only:
      print("SVD timing: {}".format(end_svd-start_svd))
    if not args.irlb_only and not args.scipy_only:
      print("Absolute error: {}".format(abserr))
    
if __name__=="__main__":
  main()

