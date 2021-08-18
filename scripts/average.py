#!/usr/bin/env python3

import sys
import numpy

def main():
    with open(sys.argv[1], "r") as f:
        data = f.readlines()
        data = [int(number.strip())for number in data]

    print("Number of datapoints:", len(data))
    print("Average:", numpy.mean(numpy.array(data)))

if __name__=="__main__":
    exit(main())
