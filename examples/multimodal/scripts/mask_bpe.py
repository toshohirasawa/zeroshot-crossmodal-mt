#!/usr/bin/env python

import sys, os
import argparse

import logging
logger = logging.getLogger(__name__)

import tokenizations

from mask import MASK_TOKEN

def main(args):
    doc = [l.strip().split(' ') for l in open(args.word)]
    doc_subword = [l.strip().split(' ') for l in open(args.subword)]
    doc_masked = [l.strip().split(' ') for l in open(args.masked_word)]
    
    assert len(doc) == len(doc_subword)

    for i, (words, subwords, words_masked) in enumerate(zip(doc, doc_subword, doc_masked)):
        try:
            alignment, _ = tokenizations.get_alignments(words, subwords)
            for i, word in enumerate(words_masked):
                if word == MASK_TOKEN:
                    for j in alignment[i]:
                        subwords[j] = MASK_TOKEN
            print(' '.join(subwords))
        except Exception as e:
            logger.info(args)
            logger.info(i)
            logger.info(words)
            logger.info(subwords)
            logger.info(words_masked)

            raise e

if __name__ == '__main__':

    # set the logging format
    # output log of at least info level to stdout
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('--word', '-w', type=str)
    parser.add_argument('--subword', '-s', type=str)
    parser.add_argument('--masked-word', '-m', type=str)

    args = parser.parse_args()

    main(args)
