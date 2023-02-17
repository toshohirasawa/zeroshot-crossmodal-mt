#!/bin/bash -eu

JA_ROOT=${1}
if [ -z ${JA_ROOT} ]; then
    echo "Usage: $0 <Japanese raw directory>"
    exit 1
fi

MOSES="./scripts/mosesdecoder/scripts/tokenizer"
export PATH="${MOSES}:$PATH"

L2=ja
BPE_MOPS=10000

# build en-ja data
echo "Load Japanese data from ${JA_ROOT}"

TOK=./data/en-${L2}/tok
BPE=./data/en-${L2}/bpe
IMG=./data/en-${L2}/image_splits
CODE=${BPE}/code
VOCAB=${BPE}/vocab

mkdir -p ${TOK} ${BPE} ${IMG}

# tok
for SPLIT in train val test_2016_flickr; do
    if [ ! -f ${TOK}/${SPLIT}.ja ]; then
        cat ${JA_ROOT}/raw/${SPLIT}.ja \
        | mecab -d /var/lib/mecab/dic/unidic -O wakati \
        | sed 's/ *$//' \
        >${TOK}/${SPLIT}.ja &

        cp ./data/en-de/tok/${SPLIT}.en ${TOK}/${SPLIT}.en &
    fi
done
wait

# bpe - joint BPE with merge operations of 10k
if [ ! -f ${VOCAB} ]; then
    subword-nmt learn-joint-bpe-and-vocab -s ${BPE_MOPS} -o ${CODE} \
        --input ${TOK}/train.en ${TOK}/train.${L2} \
        --write-vocabulary ${VOCAB}.en ${VOCAB}.${L2}

    for SPLIT in train val test_2016_flickr; do
        if [ -f ${TOK}/${SPLIT}.${L2} ]; then
            cat ${TOK}/${SPLIT}.en \
            | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB}.en \
            >${BPE}/${SPLIT}.en &

            cat ${TOK}/${SPLIT}.${L2} \
            | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB}.${L2} \
            >${BPE}/${SPLIT}.${L2} &
        fi
    done
    wait

    # bpe - shared bpe vocab
    cat ${BPE}/train.en ${BPE}/train.${L2} | python scripts/get_vocab.py >${VOCAB}
fi

# image
for SPLIT in train val test_2016_flickr; do
    if [ ! -f ${IMG}/${SPLIT}.txt ]; then
        cp ${JA_ROOT}/image_splits/${SPLIT}.txt ${IMG}/${SPLIT}.txt
    fi
done
