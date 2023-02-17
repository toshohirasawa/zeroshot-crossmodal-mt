#!/usr/bin/env python

import sys, os
import argparse

import logging
logger = logging.getLogger(__name__)

import math

def main(args):
    for line in sys.stdin:
        logit = float(line.strip())
        ppl = math.exp(-1 * logit)
        print(ppl)

if __name__ == '__main__':

    # set the logging format
    # output log of at least info level to stdout
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
