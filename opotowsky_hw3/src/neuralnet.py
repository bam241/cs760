#! /usr/bin/env python

from __future__ import print_function
from __future__ import division

import sys
import arff
import numpy as np
import pandas as pd


def main():
    # Get input: training data
    train = arff.load(open(sys.argv[1], 'rb')) 
    a = sys.argv[2]
    b = sys.argv[3]
    c = sys.argv[4]
    # data and attributes
    attr = pd.DataFrame(train['attributes'], columns=['attr', 'vars'])
    data = pd.DataFrame(train['data'], columns=train['attr'].tolist())
    
    # Print to command line
#    print('\n'.join([' '.join(v) for v in bn]))
#    print('\n', end='')
#    print('\n'.join([' '.join(v) for v in preds]))
#    print('\n', end='')
#    print(correct)
    return

if __name__ == "__main__":
    main()
