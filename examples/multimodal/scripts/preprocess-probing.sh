#!/bin/bash -eu

L2=cs
L3=fr

# tok
for MASK_TYPE in color_masking entity_masking char_masking prog_masking_5 prog_masking_10 prog_masking_15 prog_masking_20; do

    BPE=data/en_${MASK_TYPE}-${L2}_${L3}/bpe
    DATA_BIN=data-bin/en_${MASK_TYPE}-${L2}_${L3}

    # [ -d ${DATA_BIN} ] && continue

    # echo "Preprocessing en_${MASK_TYPE}-${L2}_${L3} data ..."

    # # lang_list.txt
    # mkdir -p ${DATA_BIN}
    # echo en >${DATA_BIN}/lang_list.txt
    # echo ${L2} >>${DATA_BIN}/lang_list.txt
    # echo ${L3} >>${DATA_BIN}/lang_list.txt

    # for TL in ${L2} ${L3}; do
    #     fairseq-preprocess --source-lang en --target-lang ${TL} \
    #         --trainpref ${BPE}/train.en-${TL} \
    #         --validpref ${BPE}/val.en-${TL} \
    #         --testpref ${BPE}/test_2016_flickr.en-${TL} \
    #         --srcdict ${BPE}/vocab --tgtdict ${BPE}/vocab \
    #         --thresholdsrc 0 --thresholdtgt 0 \
    #         --destdir ${DATA_BIN} \
    #         --workers 10 &
    # done
    # wait

    # features
    echo "Preprocessing feature data..."
    echo "--> en-${L2}_${L3}"

    FEAT=data/en_${MASK_TYPE}-${L2}_${L3}/feats
    DST=data-bin/en_${MASK_TYPE}-${L2}_${L3}

    for FEAT_TYPE in $(ls ${FEAT}); do
        DST_INFIX=$(echo ${FEAT_TYPE} | tr '[:upper:]' '[:lower:]')

        # train
        for TL in ${L2} ${L3}; do
        # if [ ! -f ${DST}/train.en-${TL}.${DST_INFIX}.npy ]; then
        #     ln -fs ../../${FEAT}/${FEAT_TYPE}/train.en-${TL}.npy ${DST}/train.en-${TL}.${DST_INFIX}.npy

        #     # valid - note: the SPLIT names are different
        #     ln -fs ../../${FEAT}/${FEAT_TYPE}/val.en-${TL}.npy ${DST}/valid.en-${TL}.${DST_INFIX}.npy

        #     # test - note: the SPLIT names are different
        #     ln -fs ../../${FEAT}/${FEAT_TYPE}/test_2016_flickr.en-${TL}.npy ${DST}/test.en-${TL}.${DST_INFIX}.npy

            # incongruent
            N_TEST=$(wc -l <data/en_${MASK_TYPE}-${L2}_${L3}/tok/test_2016_flickr.en-${TL}.en)
            seq 0 ${N_TEST} | head -n ${N_TEST} | tac >${DST}/test.en-${TL}.feat-incongruent.txt
        # fi
        done
    done

    # # feature padding for L3 (train, valid)
    # if [ ! -f ${DST}/train.en-${L3}.feat-padding.txt ]; then
    #     N_TRAIN=$(wc -l <data/en-${L2}_${L3}/tok/train.en-${L3}.en)
    #     N_VAL=$(wc -l <data/en-${L2}_${L3}/tok/val.en-${L3}.en)
    #     seq ${N_TRAIN} | xargs -I{} echo 1 >${DST}/train.en-${L3}.feat-padding.txt
    #     seq ${N_VAL} | xargs -I{} echo 1 >${DST}/valid.en-${L3}.feat-padding.txt
    # fi
done
