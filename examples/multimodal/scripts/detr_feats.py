#!/usr/bin/env python

import sys, os
import argparse
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
# from itertools import batched

from PIL import Image, ImageFile
from transformers import DetrImageProcessor, DetrForObjectDetection

DETR_BACKBONES = [
    'resnet-50',
    'resnet-101',
    'resnet-50-panoptic',
    'resnet-101-panoptic',
    'resnet-50-dc5',
    'resnet-101-dc5',
    'resnet-50-dc5-panoptic',
]

def basename_without_ext(path):
    return os.path.basename(path).replace('.', '_')

def remove_existing_items(src_list, dst_list):
    new_src_list = []
    new_dst_list = []

    for s, d in zip(src_list, dst_list):
        if not os.path.exists(d):
            new_src_list.append(s)
            new_dst_list.append(d)
    
    return new_src_list, new_dst_list

def main(args):
    logger.info(args)
    assert os.path.exists(args.image_dir)
    assert os.path.exists(args.image_list)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DetrImageProcessor.from_pretrained(f'facebook/detr-{args.backbone}', device=device)
    model = DetrForObjectDetection.from_pretrained(f'facebook/detr-{args.backbone}').to(device)

    image_list = [l.strip() for l in open(args.image_list)]
    input_files = [os.path.join(args.image_dir, l) for l in image_list]
    output_files =[os.path.join(args.output_dir, basename_without_ext(l) + '.npz') for l in image_list]

    input_files, output_files = remove_existing_items(input_files, output_files)

    if len(input_files) == 0:
        logger.info('All features are already extracted.')
        exit(0)

    for im, ofile in zip(tqdm(input_files), output_files):

        image = Image.open(im).convert("RGB") # make sure the mode is in RGB
        input = processor(images=image, return_tensors="pt").to(device)
        output = model(**input, output_hidden_states=True)

        feat = output.decoder_hidden_states[-1]   # 1 x 100 x 256
        feat = feat.squeeze() # 100 x 256

        feat = feat.data.cpu().float().numpy()
        np.savez_compressed(ofile, features=feat)

if __name__ == '__main__':

    # set the logging format
    # output log of at least info level to stdout
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    parser = argparse.ArgumentParser()
    
    # input json
    parser.add_argument('--image-list', help='Path of the image_list file', default=None)
    parser.add_argument('--image-dir', help='Root dir of the image path in image_list file', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--backbone', type=str, default='resnet-50', choices=DETR_BACKBONES)

    args = parser.parse_args()

    main(args)
