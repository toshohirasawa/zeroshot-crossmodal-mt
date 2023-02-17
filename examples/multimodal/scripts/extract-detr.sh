#!/bin/bash -eu

IMG_DIR=./data/images
IMG_LIST=./data/images.txt

if [ ! -d ./scripts/detr ]; then
    git clone git@github.com:facebookresearch/detr.git scripts/detr
    # use own naming scheme for each npz file
    ln -s ../detr_feats.py ./scripts/detr/detr_feats.py
fi

for BACKBONE in 'resnet-50' 'resnet-101' 'resnet-50-dc5' 'resnet-101-dc5'; do
    FEAT_DIR=./data/feats/detr_${BACKBONE}/raw
    
    # [ -d ${FEAT_DIR} ] && continue

    echo "extracting DETR (${BACKBONE}) features..."
    mkdir -p ${FEAT_DIR}

    python ./scripts/detr/detr_feats.py \
        --image-list ${IMG_LIST} \
        --image-dir ${IMG_DIR} \
        --output-dir ${FEAT_DIR} \
        --backbone ${BACKBONE}
done
