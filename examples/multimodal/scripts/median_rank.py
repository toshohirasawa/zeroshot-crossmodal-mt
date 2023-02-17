#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)

import numpy as np

def median_rank(preds: np.ndarray, golds: np.ndarray):
    if preds.ndim == 3:
        assert golds.ndim == 3
        return median_rank(preds.mean(axis=1), golds.mean(axis=1))

    assert preds.ndim == 2 and golds.ndim == 2

    sim = preds @ golds.T
    sz = preds.shape[0]

    orders = sim.argsort(axis=-1)
    ranks = orders.argsort(axis=-1)

    score = np.median(ranks[range(sz), range(sz)])

    return score

def load_np(file_or_files):
    if len(file_or_files) == 0:
        data = np.load(file_or_files[0])
    else:
        data = np.stack([np.load(file) for file in file_or_files])
    
    data = data.squeeze()
    return data

if __name__ == '__main__':

    import sys, os
    import argparse

    # set the logging format
    # output log of at least info level to stdout
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred-files', nargs='+')
    parser.add_argument('-g', '--gold-files', nargs='+')

    args = parser.parse_args()

    preds = load_np(args.pred_files)
    golds = load_np(args.gold_files)

    # print(median_rank(preds, golds))
    print(median_rank(golds, preds))
    
