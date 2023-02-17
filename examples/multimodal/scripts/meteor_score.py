#!/usr/bin/env python

import sys, os
import argparse

import logging
logger = logging.getLogger(__name__)

import evaluate

def try_get(arr, index, default):
    if len(arr) > index:
        return arr[index]
    else:
        return default
    
def main(args):

    stream = sys.stdin if args.file is None else open(args.file, 'r')

    # load stdin (should be a fairseq-generate output)
    references = []
    predictions = []
    for line in stream:
        line = line.strip()
        if line.startswith('T'):
            references.append(line.split('\t')[1])
        elif line.startswith('H'):
            predictions.append(try_get(line.split('\t'), 2, ""))
    
    assert len(references) == len(predictions)

    meteor = evaluate.load('meteor')
    score = meteor.compute(predictions=predictions, references=references)

    # scale [0, 1] to [0, 100]
    # as we usually report METEOR score in this scale.
    print(score['meteor'] * 100)

if __name__ == '__main__':

    # set the logging format
    # output log of at least info level to stdout
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, required=False)

    args = parser.parse_args()

    main(args)
