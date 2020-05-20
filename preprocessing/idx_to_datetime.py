import numpy as np
import os
from glob import glob
import parse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, type=str,
                    help='data path to either training or testing data')
parser.add_argument('--idx', type=int, help='index of radar sequence')
args = parser.parse_args()

if __name__ == '__main__':
    files = sorted([(os.path.basename(d), d) \
                    for d in glob(os.path.join(args.path, '*'))], \
                    key = lambda x: x[0])
    print(files[args.idx][0])
