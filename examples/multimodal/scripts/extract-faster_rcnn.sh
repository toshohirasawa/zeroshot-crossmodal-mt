#!/bin/bash -eu

IMG_DIR=./data/images
IMG_LIST=./data/images.txt
FEAT_DIR=./data/feats/faster_rcnn

docker pull airsplay/bottom-up-attention

MODEL_FILE=scripts/bottom-up-attention/resnet101_faster_rcnn_final_iter_320000.caffemodel
if [ ! -f ${MODEL_FILE} ]; then
    mkdir -p $(dirname ${MODEL_FILE})
    wget 'https://www.dropbox.com/s/2h4hmgcvpaewizu/resnet101_faster_rcnn_final_iter_320000.caffemodel?dl=1' -O ${MODEL_FILE}
fi

mkdir -p ${FEAT_DIR}/raw

if [ ! -f ${FEAT_DIR}/index.txt ]; then
    cp ./data/images.txt ${FEAT_DIR}/index.txt
fi

docker run --gpus all \
    -v $(realpath ./data/images):/workspace/images:ro \
    -v $(realpath ${FEAT_DIR}):/workspace/features \
    -v $(realpath ./scripts/faster_rcnn_feats.py):/opt/butd/faster_rcnn_feats.py:ro \
    -v $(realpath ./scripts/bottom-up-attention/resnet101_faster_rcnn_final_iter_320000.caffemodel):/opt/butd/resnet101_faster_rcnn_final_iter_320000.caffemodel:ro \
    --user "$(id -u):$(id -g)" --rm -it \
    --env CUDA_VISIBLE_DEVICES=0 \
    airsplay/bottom-up-attention \
    python /opt/butd/faster_rcnn_feats.py \
        --feat-dir /workspace/features/raw

# for MODEL in 'resnet50' 'resnet101'; do
#     FEAT_DIR=./data/feats/${MODEL}/raw
    
#     # [ -d ${FEAT_DIR} ] && continue

#     echo "extracting ${MODEL} features..."
#     mkdir -p ${FEAT_DIR}

#     python ./scripts/resnet_feats.py \
#         --image-list ${IMG_LIST} \
#         --image-dir ${IMG_DIR} \
#         --output-dir ${FEAT_DIR} \
#         --model ${MODEL} \
#         --batch-size 64
# done
