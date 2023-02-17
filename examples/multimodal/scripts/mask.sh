#!/bin/bash -eu

for MASK_TYPE in color char; do
    for SPLIT in train val test_2016_flickr; do
        IN_FILE=data/raw/multi30k/data/task1/tok/${SPLIT}.lc.norm.tok.en
        OUT_FILE=data/raw/multi30k/data/task1/tok/${SPLIT}.lc.norm.tok.en_${MASK_TYPE}_masking
        # if [ ! -f ${OUT_FILE} ]; then
            cat ${IN_FILE} | python ./scripts/mask.py -t ${MASK_TYPE} >${OUT_FILE} &
        # fi
    done
done
wait

for CTX in 5 10 15 20; do
    for SPLIT in train val test_2016_flickr; do
        IN_FILE=data/raw/multi30k/data/task1/tok/${SPLIT}.lc.norm.tok.en
        OUT_FILE=data/raw/multi30k/data/task1/tok/${SPLIT}.lc.norm.tok.en_prog_masking_${CTX}
        # if [ ! -f ${OUT_FILE} ]; then
            cat ${IN_FILE} | python ./scripts/mask.py -t progressive -k ${CTX} >${OUT_FILE} &
        # fi
    done
done
wait
