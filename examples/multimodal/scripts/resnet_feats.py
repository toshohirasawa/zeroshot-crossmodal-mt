#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import argparse
from pathlib import Path

import numpy as np

from PIL import Image

import torch
from torch.autograd import Variable
import torch.utils.data as data

from torchvision.models import resnet50, resnet101
from torchvision import transforms

from tqdm import tqdm

RESNET_MODELS = {
    'resnet50': resnet50,
    'resnet101': resnet101,
}

def basename_without_ext(path):
    return os.path.basename(path).replace('.', '_')

class ImageFolderDataset(data.Dataset):
    def __init__(self, im_dir, im_list, feat_dir, resize=None, crop=None):
        self.im_dir = Path(im_dir).expanduser().resolve()
        self.feat_dir = Path(feat_dir).expanduser().resolve()

        # Image list in dataset order
        self.index = Path(im_list).expanduser().resolve()

        _transforms = []
        if resize is not None:
            _transforms.append(transforms.Resize(resize))
        if crop is not None:
            _transforms.append(transforms.CenterCrop(crop))
        _transforms.append(transforms.ToTensor())
        _transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)

        self.image_files = []
        self.feat_files = []
        with self.index.open() as f:
            for fname in f:
                im_name = self.im_dir / fname.strip()
                feat_name = self.feat_dir / (basename_without_ext(fname.strip()) + '.npz')
                if not feat_name.exists():
                    assert im_name.exists(), "{} does not exist.".format(im_name)
                    self.image_files.append(str(im_name))
                    self.feat_files.append(str(feat_name))

    def read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)

    def __getitem__(self, idx):
        return {
            'im': self.read_image(self.image_files[idx]),
            'name': self.image_files[idx],
            'feat_name': self.feat_files[idx],
        }

    def __len__(self):
        return len(self.image_files)


def resnet_forward(cnn, x):
    x = cnn.conv1(x)
    x = cnn.bn1(x)
    x = cnn.relu(x)
    x = cnn.maxpool(x)

    x = cnn.layer1(x)
    x = cnn.layer2(x)
    res4f_relu = cnn.layer3(x)
    res5e_relu = cnn.layer4(res4f_relu)

    avgp = cnn.avgpool(res5e_relu)
    avgp = avgp.view(avgp.size(0), -1)
    return res4f_relu.data.cpu().float().numpy(), avgp.data.cpu().float().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-list', help='Path of the image_list file', default=None)
    parser.add_argument('--image-dir', help='Root dir of the image path in image_list file', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--model', type=str, default='resnet50', choices=RESNET_MODELS.keys())
    parser.add_argument('--batch-size', type=int, default=256)

    # Parse arguments
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset
    dataset = ImageFolderDataset(args.image_dir, args.image_list, args.output_dir, resize=256, crop=224)
    if len(dataset) == 0:
        print('All features are already extracted.')
        exit(0)

    loader = data.DataLoader(dataset, batch_size=args.batch_size)

    print('Creating CNN instance.')
    cnn = RESNET_MODELS[args.model](pretrained=True).to(device)
    cnn.eval()

    for bidx, batch in tqdm(enumerate(loader)):
        feat_names = batch['feat_name']
        x = Variable(batch['im'], volatile=True).to(device)
        feats, g_feats = resnet_forward(cnn, x)

        for i, feat_name in enumerate(feat_names):
            np.savez_compressed(feat_name, 
                features=feats[i],
                g_feature=g_feats[i],
            )
