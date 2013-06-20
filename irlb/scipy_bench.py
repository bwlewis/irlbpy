from __future__ import division, print_function
from irlb import *
from argparse import *

import timeit

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
  return parser.parse_args()

def main():
  args = parse_args()
  A=np.random.rand(args.num_row*args.num_col).reshape(args.num_row,args.num_col)

  start_irlb = timeit.timeit()
  X = irlb(A, args.nu, args.tol)
  end_irlb = timeit.timeit()

  # Compare estimated values with np.linalg.svd:
  start_svd = timeit.timeit()
  S = np.linalg.svd(A, 0)
  end_svd = timeit.timeit()

  abserr = np.max(np.abs(X[1]-S[1][0:args.nu]))

  if (args.csv):
    print("{},{},{}".format(end_irlb-start_irlb, end_svd-start_svd, abserr))
  else:
    print("IRLB timing: {}".format(end_irlb-start_irlb))
    print("SVD timing: {}".format(end_svd-start_svd))
    print("Absolute error: {}".format(abserr))
    
if __name__=="__main__":
  main()

