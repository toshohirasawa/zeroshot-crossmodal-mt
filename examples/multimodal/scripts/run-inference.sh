#!/bin/bash -eu

GPU=${1:-0}
CKPT_ROOT=${2:-checkpoint}

for CKPT in $(find ${CKPT_ROOT} -name "averaged_model.pt"); do
    TASK=$(echo ${CKPT} | cut -d '/' -f 2)
    LANG_PAIR=$(echo ${CKPT} | cut -d '/' -f 3)
    FEAT_TYPE=$(echo ${CKPT} | cut -d '/' -f 4)
    MODEL=$(echo ${CKPT} | cut -d '/' -f 5)
    SEED=$(echo ${CKPT} | cut -d '/' -f 6)
    # echo ${LANG_PAIR} ${MODEL}.sh ${SEED}

    bash model/${TASK}/${LANG_PAIR}/${FEAT_TYPE}/${MODEL}.sh ${GPU} ${SEED}
done
