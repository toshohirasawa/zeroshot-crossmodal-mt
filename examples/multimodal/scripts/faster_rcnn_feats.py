# !/usr/bin/env python

# The root of bottom-up-attention repo. Do not need to change if using provided docker file.
BUTD_ROOT = '/opt/butd/'

import os, sys
sys.path.insert(0, BUTD_ROOT + "/tools")
os.environ['GLOG_minloglevel'] = '2'

import logging
logger = logging.getLogger(__name__)

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms

import caffe
import argparse
import pprint
import base64
import numpy as np
import cv2
import csv
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36

def basename_without_ext(path):
    return os.path.basename(path).replace('.', '_')

def load_image_and_feat_files(img_root, feat_root, img_list):
    names = [l.strip() for l in open(img_list)]
    image_files = [os.path.join(img_root, n) for n in names]
    feat_files = [os.path.join(feat_root, basename_without_ext(n) + '.npz') for n in names]

    return image_files, feat_files

def remove_existing_items(src_list, dst_list):
    new_src_list = []
    new_dst_list = []

    for s, d in zip(src_list, dst_list):
        if not os.path.exists(d):
            new_src_list.append(s)
            new_dst_list.append(d)
    
    return new_src_list, new_dst_list

def generate_feats(prototxt, weights, image_files, feat_files):
    missing_image_files, missing_feat_files = remove_existing_items(image_files, feat_files)

    if len(image_files) == 0:
        logger.info('already completed {:d}'.format(len(image_files)))
    else:
        logger.info('generating {:d}/{:d}'.format(len(missing_image_files), len(image_files)))

        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        for im_file, feat_file in tqdm(zip(missing_image_files, missing_feat_files), total=len(missing_image_files)):
            feat_dict = get_detections_from_im(net, im_file)
            np.savez_compressed(feat_file, **feat_dict)

def get_detections_from_im(net, im_file, conf_thresh=0.2):
    """
    :param net:
    :param im_file: full path to an image
    :param conf_thresh:
    :return: all information from detection and attr prediction
    """
    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    objects = np.argmax(cls_prob[keep_boxes][:, 1:], axis=1)
    objects_conf = np.max(cls_prob[keep_boxes][:, 1:], axis=1)
    attrs = np.argmax(attr_prob[keep_boxes][:, 1:], axis=1)
    attrs_conf = np.max(attr_prob[keep_boxes][:, 1:], axis=1)

    return {
        "img_h": np.size(im, 0),
        "img_w": np.size(im, 1),
        "objects_id": base64.b64encode(objects),  # int64
        "objects_conf": base64.b64encode(objects_conf),  # float32
        "attrs_id": base64.b64encode(attrs),  # int64
        "attrs_conf": base64.b64encode(attrs_conf),  # float32
        "num_boxes": len(keep_boxes),
        "boxes": base64.b64encode(cls_boxes[keep_boxes]),  # float32
        "features": base64.b64encode(pool5[keep_boxes])  # float32
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')

    # leave those arguments default
    parser.add_argument('--def', dest='prototxt', default=None, type=str, 
                        help='prototxt file defining the network')
    parser.add_argument('--cfg', dest='cfg_file', default=None, type=str, 
                        help='optional config file')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set config keys')
    parser.add_argument('--caffemodel', type=str, 
                        default='/opt/butd/resnet101_faster_rcnn_final_iter_320000.caffemodel')
    parser.add_argument('--image-dir', type=str, default='/workspace/images/')
    parser.add_argument('--image-list', type=str, default='/workspace/features/index.txt')
    parser.add_argument('--feat-dir', type=str, default='/workspace/features')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # set the logging format
    # output log of at least info level to stdout
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()

    args.cfg_file = BUTD_ROOT + "experiments/cfgs/faster_rcnn_end2end_resnet.yml" # s = 500
    args.prototxt = BUTD_ROOT + "models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt"
    
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    # Load image ids, need modification for new datasets.
    im_files, feat_files = load_image_and_feat_files(args.image_dir, args.feat_dir, args.image_list)  
    
    # Generate TSV files, noramlly do not need to modify
    generate_feats(args.prototxt, args.caffemodel, im_files, feat_files)
