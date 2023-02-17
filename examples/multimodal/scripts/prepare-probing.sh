#!/bin/bash -eu
export LC_ALL=en_US.UTF_8

echo "generating data..."

# en-{de,fr,cs} from multi30k
M30K=./data/raw/multi30k/data/task1
REL_M30K=./../../raw/multi30k/data/task1
BPE_MOPS=10000

L2=cs
L3=fr

for MASK_TYPE in color_masking entity_masking char_masking prog_masking_5 prog_masking_10 prog_masking_15 prog_masking_20; do

    TOK=./data/en_${MASK_TYPE}-${L2}_${L3}/tok
    BPE=./data/en_${MASK_TYPE}-${L2}_${L3}/bpe

    CODE=${BPE}/code
    VOCAB=${BPE}/vocab

    TOK_ORG=./data/en-${L2}_${L3}/tok
    BPE_ORG=./data/en-${L2}_${L3}/bpe

    [ -d ${TOK} ] && continue
    
    mkdir -p ${TOK} ${BPE}

    # tok
    head -n14500 ${M30K}/tok/train.lc.norm.tok.en_${MASK_TYPE} >${TOK}/train.en-${L2}.en
    tail -n14500 ${M30K}/tok/train.lc.norm.tok.en_${MASK_TYPE} >${TOK}/train.en-${L3}.en
    ln -s ../../en-${L2}_${L3}/tok/train.en-${L2}.${L2} ${TOK}/train.en-${L2}${L2}
    ln -s ../../en-${L2}_${L3}/tok/train.en-${L3}.${L3} ${TOK}/train.en-${L3}${L3}

    ln -s ${REL_M30K}/tok/val.lc.norm.tok.en_${MASK_TYPE} ${TOK}/val.en-${L2}.en
    ln -s ${REL_M30K}/tok/val.lc.norm.tok.en_${MASK_TYPE} ${TOK}/val.en-${L3}.en
    ln -s ../../en-${L2}_${L3}/tok/val.en-${L2}.${L2} ${TOK}/val.en-${L2}.${L2}
    ln -s ../../en-${L2}_${L3}/tok/val.en-${L3}.${L3} ${TOK}/val.en-${L3}.${L3}

    ln -s ${REL_M30K}/tok/test_2016_flickr.lc.norm.tok.en_${MASK_TYPE} ${TOK}/test_2016_flickr.en-${L2}.en
    ln -s ${REL_M30K}/tok/test_2016_flickr.lc.norm.tok.en_${MASK_TYPE} ${TOK}/test_2016_flickr.en-${L3}.en
    ln -s ../../en-${L2}_${L3}/tok/test_2016_flickr.en-${L2}.${L2} ${TOK}/test_2016_flickr.en-${L2}.${L2}
    ln -s ../../en-${L2}_${L3}/tok/test_2016_flickr.en-${L3}.${L3} ${TOK}/test_2016_flickr.en-${L3}.${L3}

    # images and feats
    ln -fs ../en-cs_fr/image_splits ./data/en_${MASK_TYPE}-${L2}_${L3}/
    ln -fs ../en-cs_fr/feats ./data/en_${MASK_TYPE}-${L2}_${L3}/

    # bpe
    for TL in ${L2} ${L3}; do
    for SPLIT in train val test_2016_flickr; do
        python ./scripts/mask_bpe.py \
            -w ${TOK_ORG}/${SPLIT}.en-${TL}.en \
            -s ${BPE_ORG}/${SPLIT}.en-${TL}.en \
            -m ${TOK}/${SPLIT}.en-${TL}.en \
        >${BPE}/${SPLIT}.en-${TL}.en &
        ln -s ../../en-${L2}_${L3}/tok/${SPLIT}.en-${TL}.${TL} ${BPE}/${SPLIT}.en-${TL}.${TL} &
    done
    done
    wait

    ln -s ../../en-${L2}_${L3}/bpe/vocab ${BPE}/vocab
done

# for L2 in de fr cs tr; do
# for L3 in de fr cs tr; do
#     [ "${L2}" == "${L3}" ] && continue
#     echo "-> en-${L2}_${L3}"


#     [ -d ${TOK} ] && continue
    
#     mkdir -p ${TOK} ${BPE} ${IMG}

#     # tok - train
#     head -n14500 ${TOK_L2}/train.en    >${TOK}/train.en-${L2}.en
#     head -n14500 ${TOK_L2}/train.${L2} >${TOK}/train.en-${L2}.${L2}
#     tail -n14500 ${TOK_L3}/train.en    >${TOK}/train.en-${L3}.en
#     tail -n14500 ${TOK_L3}/train.${L3} >${TOK}/train.en-${L3}.${L3}

#     # tok - val
#     ln -s ./../../en-${L2}/tok/val.en    ${TOK}/val.en-${L2}.en
#     ln -s ./../../en-${L2}/tok/val.${L2} ${TOK}/val.en-${L2}.${L2}
#     ln -s ./../../en-${L3}/tok/val.en    ${TOK}/val.en-${L3}.en
#     ln -s ./../../en-${L3}/tok/val.${L3} ${TOK}/val.en-${L3}.${L3}

#     # tok - test
#     for SPLIT in test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr test_ambigcaps test_commute test_commute_incorrect; do
#         if [ -f ${TOK_L2}/${SPLIT}.${L2} ]; then
#             ln -s ./../../en-${L2}/tok/${SPLIT}.en    ${TOK}/${SPLIT}.en-${L2}.en
#             ln -s ./../../en-${L2}/tok/${SPLIT}.${L2} ${TOK}/${SPLIT}.en-${L2}.${L2}
#         fi
#         if [ -f ${TOK_L3}/${SPLIT}.${L3} ]; then
#             ln -s ./../../en-${L3}/tok/${SPLIT}.en    ${TOK}/${SPLIT}.en-${L3}.en
#             ln -s ./../../en-${L3}/tok/${SPLIT}.${L3} ${TOK}/${SPLIT}.en-${L3}.${L3}
#         fi
#     done

#     # image splits
#     head -n14500 ${IMG_L2}/train.txt >${IMG}/train.en-${L2}.txt
#     tail -n14500 ${IMG_L3}/train.txt >${IMG}/train.en-${L3}.txt
#     ln -s ./../../en-${L2}/image_splits/val.txt ${IMG}/val.en-${L2}.txt
#     ln -s ./../../en-${L3}/image_splits/val.txt ${IMG}/val.en-${L3}.txt
#     for SPLIT in test test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr test_ambigcaps test_commute test_commute_incorrect; do
#         if [ -f ${IMG_L2}/${SPLIT}.txt ]; then
#             ln -s ./../../en-${L2}/image_splits/${SPLIT}.txt  ${IMG}/${SPLIT}.en-${L2}.txt
#         fi
#         if [ -f ${IMG_L3}/${SPLIT}.txt ]; then
#             ln -s ./../../en-${L3}/image_splits/${SPLIT}.txt  ${IMG}/${SPLIT}.en-${L3}.txt
#         fi
#     done

#     # bpe - code
#     cat ${TOK}/train.en-${L2}.en ${TOK}/train.en-${L3}.en ${TOK}/train.en-${L2}.${L2} ${TOK}/train.en-${L3}.${L3} \
#     | subword-nmt learn-bpe -s ${BPE_MOPS} >${CODE}
    
#     # bpe - vocab
#     cat ${TOK}/train.en-${L2}.en ${TOK}/train.en-${L3}.en ${TOK}/train.en-${L2}.${L2} ${TOK}/train.en-${L3}.${L3} \
#     | subword-nmt apply-bpe -c ${CODE} \
#     | subword-nmt get-vocab >${VOCAB}

#     # bpe - apply
#     for SPLIT in train val test test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr test_ambigcaps test_commute test_commute_incorrect; do
#     for TL in ${L2} ${L3}; do
#         if [ -f ${TOK}/${SPLIT}.en-${TL}.${TL} ]; then
#             cat ${TOK}/${SPLIT}.en-${TL}.en \
#             | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB} --vocabulary-threshold 1 \
#             >${BPE}/${SPLIT}.en-${TL}.en &

#             cat ${TOK}/${SPLIT}.en-${TL}.${TL} \
#             | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB} --vocabulary-threshold 1 \
#             >${BPE}/${SPLIT}.en-${TL}.${TL} &
#         fi
#     done
#     done
#     wait
# done
# done