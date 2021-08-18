#!/usr/bin/env python3

# Input: output of `awk '{print length($0)}' FILENAME > LINEFILE`
#        each line is the length of the nth line of FILENAME
# Usage: python3 count.py LINEFILE

import sys
import numpy as np

if __name__=="__main__":
    length_file = sys.argv[1]
    with open(length_file, 'r') as f:
        lengths = f.readlines()
    lengths = np.array([int(length.strip()) for length in lengths])
    #np.argmax(lengths)
    lengths = np.sort(lengths)
    print('Average:', np.mean(lengths))
    print('Max 10:', lengths[-10:])
    
