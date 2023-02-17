#!/usr/bin/env python

import sys, os
import argparse

import logging
logger = logging.getLogger(__name__)

from comet import download_model, load_from_checkpoint

def try_get(arr, index, default):
    if len(arr) > index:
        return arr[index]
    else:
        return default

def main(args):

    stream = sys.stdin if args.file is None else open(args.file, 'r')

    # load stdin (should be a fairseq-generate output)
    sources = []
    references = []
    predictions = []
    for line in stream:
        line = line.strip()
        if line.startswith('T'):
            references.append(line.split('\t')[1])
        elif line.startswith('H'):
            predictions.append(try_get(line.split('\t'), 2, ""))
        elif line.startswith('S'):
            sources.append(line.split('\t')[1])

    assert len(references) == len(predictions)
    assert len(references) == len(sources)
    
    logger.info(f'loaded {len(references)} items')

    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet = load_from_checkpoint(comet_path)

    eval_data = [
        {
            'src': s,
            'mt': p,
            'ref': r
        }
        for s, p, r in zip(sources, references, predictions)
    ]
    
    score = comet.predict(eval_data, batch_size=args.batch_size, gpus=args.gpus)

    print(score[1])

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
    parser.add_argument("--batch-size", type=int, default=8, required=False)
    parser.add_argument('--gpus', type=int, default=0, required=False)

    args = parser.parse_args()

    main(args)
