#!/bin/bash -eu

# tok
for L2 in de fr cs tr ja; do
for L3 in de fr cs tr ja; do
    [ "${L2}" == "${L3}" ] && continue
    [ ! -d data/en-${L2}_${L3} ] && continue

    TEXT=data/en-${L2}_${L3}/bpe
    DATA=data-bin/en-${L2}_${L3}/

    [ -d ${DATA} ] && continue

    echo "Preprocessing en-${L2}_${L3} data ..."

    # lang_list.txt
    mkdir -p data-bin/en-${L2}_${L3}
    echo en >data-bin/en-${L2}_${L3}/lang_list.txt
    echo ${L2} >>data-bin/en-${L2}_${L3}/lang_list.txt
    echo ${L3} >>data-bin/en-${L2}_${L3}/lang_list.txt

    # en-L2
    TEST_L2=data-bin/en-${L2}_${L3}/test_list.en-${L2}.txt
    ls data/en-${L2}_${L3}/bpe/test_*.${L2} | \
        xargs -I{} basename {} .${L2} | \
        sort | \
        xargs -I{} echo ${TEXT}/{} >${TEST_L2}
    
    fairseq-preprocess --source-lang en --target-lang ${L2} \
        --trainpref ${TEXT}/train.en-${L2} \
        --validpref ${TEXT}/val.en-${L2} \
        --testpref $(paste -s -d ',' ${TEST_L2}) \
        --srcdict ${TEXT}/vocab --tgtdict ${TEXT}/vocab \
        --thresholdsrc 0 --thresholdtgt 0 \
        --destdir ${DATA} \
        --workers 10 &

    # en-L3
    TEST_L3=data-bin/en-${L2}_${L3}/test_list.en-${L3}.txt
    ls data/en-${L2}_${L3}/bpe/test_*.${L3} | \
        xargs -I{} basename {} .${L3} | \
        sort | \
        xargs -I{} echo ${TEXT}/{} >${TEST_L3}

    fairseq-preprocess --source-lang en --target-lang ${L3} \
        --trainpref ${TEXT}/train.en-${L3} \
        --validpref ${TEXT}/val.en-${L3} \
        --testpref $(paste -s -d ',' ${TEST_L3}) \
        --srcdict ${TEXT}/vocab --tgtdict ${TEXT}/vocab \
        --thresholdsrc 0 --thresholdtgt 0 \
        --destdir ${DATA} \
        --workers 10 &

    wait
done
done

# features
echo "Preprocessing feature data..."
# for L2 in de fr cs tr ja; do
# for L3 in de fr cs tr ja; do
for L2 in cs; do
for L3 in fr; do
    [ "${L2}" == "${L3}" ] && continue
    [ ! -d data/en-${L2}_${L3} ] && continue
    
    echo "--> en-${L2}_${L3}"

    FEAT=data/en-${L2}_${L3}/feats
    DST=data-bin/en-${L2}_${L3}

    mkdir -p ${DST}

    for FEAT_TYPE in $(ls ${FEAT}); do
        DST_INFIX=$(echo ${FEAT_TYPE} | tr '[:upper:]' '[:lower:]')

        # train
        if [ ! -f ${DST}/train.en-${L2}.${DST_INFIX}.npy ]; then
            ln -fs ../../${FEAT}/${FEAT_TYPE}/train.en-${L2}.npy ${DST}/train.en-${L2}.${DST_INFIX}.npy
            ln -fs ../../${FEAT}/${FEAT_TYPE}/train.en-${L3}.npy ${DST}/train.en-${L3}.${DST_INFIX}.npy

            # valid - note: the SPLIT names are different
            ln -fs ../../${FEAT}/${FEAT_TYPE}/val.en-${L2}.npy ${DST}/valid.en-${L2}.${DST_INFIX}.npy
            ln -fs ../../${FEAT}/${FEAT_TYPE}/val.en-${L3}.npy ${DST}/valid.en-${L3}.${DST_INFIX}.npy
        fi

        # test
        for TL in ${L2} ${L3}; do
            TEST_NO=0
            for SPLIT in $(cat data-bin/en-${L2}_${L3}/test_list.en-${TL}.txt | xargs -I{} basename {} .en-${TL}); do
                if [ ${TEST_NO} == 0 ]; then
                    TEST_NAME=test
                else
                    TEST_NAME=test${TEST_NO}
                fi
                
                ln -fs ../../${FEAT}/${FEAT_TYPE}/${SPLIT}.en-${TL}.npy ${DST}/${TEST_NAME}.en-${TL}.${DST_INFIX}.npy

                # incongruent
                N_TEST=$(wc -l <data/en-${L2}_${L3}/tok/${SPLIT}.en-${TL}.en)
                seq 0 ${N_TEST} | head -n ${N_TEST} | tac >${DST}/${TEST_NAME}.en-${TL}.feat-incongruent.txt

                TEST_NO=$((TEST_NO+1))
            done
        done
    done

    # feature padding for L3 (train, valid)
    if [ ! -f ${DST}/train.en-${L3}.feat-padding.txt ]; then
        N_TRAIN=$(wc -l <data/en-${L2}_${L3}/tok/train.en-${L3}.en)
        N_VAL=$(wc -l <data/en-${L2}_${L3}/tok/val.en-${L3}.en)
        seq ${N_TRAIN} | xargs -I{} echo 1 >${DST}/train.en-${L3}.feat-padding.txt
        seq ${N_VAL} | xargs -I{} echo 1 >${DST}/valid.en-${L3}.feat-padding.txt
    fi
done
done
