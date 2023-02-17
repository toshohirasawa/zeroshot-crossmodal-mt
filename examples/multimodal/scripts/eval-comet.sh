#!/bin/bash -eu

FILE=${1}
GPU=${2:-0}

STATE=$(grep "COMET = " ${FILE} | wc -l)
if [ $STATE -eq 0 ]; then
    SCORE=$(CUDA_VISIBLE_DEVICES=${GPU} python scripts/comet_score.py --file ${FILE} --gpus 1)
    echo "COMET = ${SCORE}" 1>>${FILE}
fi
