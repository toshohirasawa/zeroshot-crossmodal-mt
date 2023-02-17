#!/usr/bin/env python

import sys, os
import argparse
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

def main(args):
    vocab = defaultdict(int)
    for line in sys.stdin:
        for tok in line.strip().split():
            if len(tok) > 0:
                vocab[tok] += 1
    
    for tok, freq in sorted(vocab.items(), key=lambda i: i[1], reverse=True):
        print(f'{tok} {freq}')

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