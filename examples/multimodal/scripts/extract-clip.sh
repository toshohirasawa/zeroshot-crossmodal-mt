#!/bin/bash -eu

IMG_DIR=./data/images
IMG_LIST=./data/images.txt

if [ ! -d ./scripts/OpenAI-CLIP-Feature ]; then
    git clone git@github.com:jianjieluo/OpenAI-CLIP-Feature.git scripts/OpenAI-CLIP-Feature
    # use own naming scheme for each npz file
    ln -fs ../clip_visual_feats.py ./scripts/OpenAI-CLIP-Feature/clip_visual_feats.py
fi

for VE_NAME in RN101 ViT-B/32 ViT-B/16 RN101_448 ViT-B/32_448; do
    FEAT_DIR=./data/feats/clip_$(echo ${VE_NAME} | sed -e 's/\//-/')/raw
    
    # [ -d ${FEAT_DIR} ] && continue

    echo "Extracting CLIP features for ${IMG_LIST} by ${VE_NAME}"
    mkdir -p ${FEAT_DIR}

    MODEL=$(echo ${VE_NAME} | cut -d"_" -f1)

    python3 ./scripts/OpenAI-CLIP-Feature/clip_visual_feats.py \
        --image_list ${IMG_LIST} \
        --image_dir ${IMG_DIR} \
        --output_dir ${FEAT_DIR} \
        --ve_name ${VE_NAME} \
        --model_type_or_path ${MODEL}
done