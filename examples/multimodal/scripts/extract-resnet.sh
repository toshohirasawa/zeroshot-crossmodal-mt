#!/bin/bash -eu

IMG_DIR=./data/images
IMG_LIST=./data/images.txt

for MODEL in 'resnet50' 'resnet101'; do
    FEAT_DIR=./data/feats/${MODEL}/raw
    
    # [ -d ${FEAT_DIR} ] && continue

    echo "extracting ${MODEL} features..."
    mkdir -p ${FEAT_DIR}

    python ./scripts/resnet_feats.py \
        --image-list ${IMG_LIST} \
        --image-dir ${IMG_DIR} \
        --output-dir ${FEAT_DIR} \
        --model ${MODEL} \
        --batch-size 64
done
