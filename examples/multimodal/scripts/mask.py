#!/usr/bin/env python

import sys, os
import argparse

import logging
logger = logging.getLogger(__name__)

from matplotlib.colors import is_color_like

MASK_TOKEN = '[v]'

def get_color_filter(**kwargs):
    def color_filter(i:int, token:str, **kwargs):
        return is_color_like(str(token))

    return color_filter

def get_char_filter(char_words=["man", "woman", "people", "men", "girl", "boy",], **kwargs):
    def char_filter(i:int, token:str):
        return str(token) in char_words

    return char_filter
    
def get_progressive_filter(context_window:int, **kwargs):
    assert context_window is not None

    def prog_filter(i:int, token:str):
        return i >= context_window

    return prog_filter

def get_entity_filter(**kwargs):
    pass

FILTERS = {
    'color': get_color_filter,
    'character': get_char_filter,
    'char': get_char_filter,
    'progressive': get_progressive_filter,
    'prog': get_progressive_filter,
}

def main(args):
    # parser = spacy.load("en_core_web_lg")
    is_masked = FILTERS[args.mask_type](**args.__dict__)

    for line in sys.stdin:
        tokens = line.strip().split(' ')

        masked = [MASK_TOKEN if is_masked(i, t) else str(t) for i, t in enumerate(tokens)]

        print(' '.join(masked))

if __name__ == '__main__':

    # set the logging format
    # output log of at least info level to stdout
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask-type', '-t', type=str, choices=list(FILTERS.keys()), required=True)
    parser.add_argument('--context-window', '-k', type=int, default=None)

    args = parser.parse_args()

    main(args)
