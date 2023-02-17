#!/usr/bin/env python

import sys, os
import argparse

import logging
logger = logging.getLogger(__name__)

import langdetect

def try_get(arr, index, default):
    if len(arr) > index:
        return arr[index]
    else:
        return default
    
def try_detect_lang(tok):
    try:
        return langdetect.detect(tok)
    except langdetect.LangDetectException:
        return "n/a"
    except:
        return "error"

def main(args):
    for line in sys.stdin:
        if not line.startswith("H"):
            continue

        toks = try_get(line.strip().split('\t'), 2, "").split()
        langs = [try_detect_lang(t) for t in toks]
        print(' '.join(langs))

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
