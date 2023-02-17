#!/bin/bash -eu
export LC_ALL=en_US.UTF_8

BPE_MOPS=10000

if [ ! -d ./data/en-de ]; then
    echo "run scripts/prepare-data.sh first"
    exit 1
fi

# tok
echo "generating multilingual data..."
for L2 in de fr cs tr; do
for L3 in de fr cs tr; do
    [ "${L2}" == "${L3}" ] && continue
    [ ! -d ./data/en-${L2} ] && continue
    [ ! -d ./data/en-${L3} ] && continue

    echo "-> en-${L2}_${L3}"

    TOK=./data/en-${L2}_${L3}/tok
    BPE=./data/en-${L2}_${L3}/bpe
    IMG=./data/en-${L2}_${L3}/image_splits

    CODE=${BPE}/code
    VOCAB=${BPE}/vocab

    TOK_L2=./data/en-${L2}/tok
    TOK_L3=./data/en-${L3}/tok
    IMG_L2=./data/en-${L2}/image_splits
    IMG_L3=./data/en-${L3}/image_splits

    [ -d ${TOK} ] && continue
    
    mkdir -p ${TOK} ${BPE} ${IMG}

    # tok - train
    head -n14500 ${TOK_L2}/train.en    >${TOK}/train.en-${L2}.en
    head -n14500 ${TOK_L2}/train.${L2} >${TOK}/train.en-${L2}.${L2}
    tail -n14500 ${TOK_L3}/train.en    >${TOK}/train.en-${L3}.en
    tail -n14500 ${TOK_L3}/train.${L3} >${TOK}/train.en-${L3}.${L3}

    # tok - val
    ln -s ./../../en-${L2}/tok/val.en    ${TOK}/val.en-${L2}.en
    ln -s ./../../en-${L2}/tok/val.${L2} ${TOK}/val.en-${L2}.${L2}
    ln -s ./../../en-${L3}/tok/val.en    ${TOK}/val.en-${L3}.en
    ln -s ./../../en-${L3}/tok/val.${L3} ${TOK}/val.en-${L3}.${L3}

    # tok - test
    for SPLIT in test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr test_ambigcaps test_commute test_commute_incorrect; do
        if [ -f ${TOK_L2}/${SPLIT}.${L2} ]; then
            ln -s ./../../en-${L2}/tok/${SPLIT}.en    ${TOK}/${SPLIT}.en-${L2}.en
            ln -s ./../../en-${L2}/tok/${SPLIT}.${L2} ${TOK}/${SPLIT}.en-${L2}.${L2}
        fi
        if [ -f ${TOK_L3}/${SPLIT}.${L3} ]; then
            ln -s ./../../en-${L3}/tok/${SPLIT}.en    ${TOK}/${SPLIT}.en-${L3}.en
            ln -s ./../../en-${L3}/tok/${SPLIT}.${L3} ${TOK}/${SPLIT}.en-${L3}.${L3}
        fi
    done

    # image splits
    head -n14500 ${IMG_L2}/train.txt >${IMG}/train.en-${L2}.txt
    tail -n14500 ${IMG_L3}/train.txt >${IMG}/train.en-${L3}.txt
    ln -s ./../../en-${L2}/image_splits/val.txt ${IMG}/val.en-${L2}.txt
    ln -s ./../../en-${L3}/image_splits/val.txt ${IMG}/val.en-${L3}.txt
    for SPLIT in test test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr test_ambigcaps test_commute test_commute_incorrect; do
        if [ -f ${IMG_L2}/${SPLIT}.txt ]; then
            ln -s ./../../en-${L2}/image_splits/${SPLIT}.txt  ${IMG}/${SPLIT}.en-${L2}.txt
        fi
        if [ -f ${IMG_L3}/${SPLIT}.txt ]; then
            ln -s ./../../en-${L3}/image_splits/${SPLIT}.txt  ${IMG}/${SPLIT}.en-${L3}.txt
        fi
    done

    # bpe - code
    cat ${TOK}/train.en-${L2}.en ${TOK}/train.en-${L3}.en ${TOK}/train.en-${L2}.${L2} ${TOK}/train.en-${L3}.${L3} \
    | subword-nmt learn-bpe -s ${BPE_MOPS} >${CODE}
    
    # bpe - vocab
    cat ${TOK}/train.en-${L2}.en ${TOK}/train.en-${L3}.en ${TOK}/train.en-${L2}.${L2} ${TOK}/train.en-${L3}.${L3} \
    | subword-nmt apply-bpe -c ${CODE} \
    | subword-nmt get-vocab >${VOCAB}

    # bpe - apply
    for SPLIT in train val test test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr test_ambigcaps test_commute test_commute_incorrect; do
    for TL in ${L2} ${L3}; do
        if [ -f ${TOK}/${SPLIT}.en-${TL}.${TL} ]; then
            cat ${TOK}/${SPLIT}.en-${TL}.en \
            | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB} --vocabulary-threshold 1 \
            >${BPE}/${SPLIT}.en-${TL}.en &

            cat ${TOK}/${SPLIT}.en-${TL}.${TL} \
            | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB} --vocabulary-threshold 1 \
            >${BPE}/${SPLIT}.en-${TL}.${TL} &
        fi
    done
    done
    wait
done
done