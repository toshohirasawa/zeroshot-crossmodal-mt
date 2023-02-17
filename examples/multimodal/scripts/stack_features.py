#!/usr/bin/env python

# output shape is (n_data, channel, n_feat)

import sys, os
import argparse

import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm
import base64
import numpy as np

def im_name_to_npz_name(path):
    return os.path.basename(path).split("#")[0].replace('.', '_') + '.npz'

def decode_faster_rcnn(x):
    decoded = base64.b64decode(x)
    feat = np.frombuffer(decoded, dtype=np.float32).reshape(-1, 2048).transpose()

    return feat

FEATURE_TYPES = {
    'resnet': lambda x: x.reshape(x.shape[0], -1),
    'clip': lambda x: x.transpose(),
    'detr': lambda x: x.transpose(),
    'faster_rcnn': decode_faster_rcnn,
}

def load_features(file, key):
    try:
        return np.load(file)[key]
    except Exception as e:
        print(f'Cannot load {file}')
        raise e

def main(args):
    logger.info(args)

    assert not os.path.exists(args.output), \
        f'Stack file already exists: {args.output}'
    
    
    transform = FEATURE_TYPES[args.feat_type]

    feat_files = [os.path.join(args.feat_dir, im_name_to_npz_name(f.strip())) for f in open(args.image_list)]
    assert all([os.path.exists(f) for f in feat_files]), \
        'Missing feature files (displayed up to 10 items):\n{}'.format('\n'.join(
            [f for f in feat_files if not os.path.exists(f)][:10]
        ))

    feats = [transform(load_features(f, args.feat_key)) for f in tqdm(feat_files)]

    out_feat = np.stack(feats)
    np.save(args.output, out_feat)

    logger.info(f"Saved features of {out_feat.shape}: {args.output}")

if __name__ == '__main__':

    # set the logging format
    # output log of at least info level to stdout
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-dir', type=str)
    parser.add_argument('--image-list', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--feat-type', type=str, choices=list(FEATURE_TYPES.keys()))

    parser.add_argument('--feat-key', type=str, default='features')

    args = parser.parse_args()

    main(args)
