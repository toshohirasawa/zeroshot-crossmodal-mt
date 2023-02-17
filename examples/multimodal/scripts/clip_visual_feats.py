from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import visual_extractor
import basic_utils as utils
from params import parse_args


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
    # load image list
    imgs = utils.read_lines(args.image_list)

    # build src_list & dst_list
    src_list = [os.path.join(args.image_dir, img) for img in imgs]
    dst_list = [os.path.join(args.output_dir, basename_without_ext(img) + '.npz') for img in imgs]
    
    src_list, dst_list = remove_existing_items(src_list, dst_list)
    if len(src_list) == 0:
        print('All features are already extracted.')
        exit(0)

    # extract
    worker = visual_extractor.create(args.ve_name, args, src_list, dst_list)
    worker.extract()

if __name__ == "__main__":
    args = parse_args()
    if not args.debug:
        utils.mkdirp(args.output_dir)
    main(args)
