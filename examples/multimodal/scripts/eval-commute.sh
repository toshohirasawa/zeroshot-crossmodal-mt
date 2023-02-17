#!/bin/bash -eu

CORRECT=test4
INCORRECT=test5

CKPT=checkpoint/zero-shot/en-cs_fr/detr_resnet-50-dc5/imagination-lvp_static
for SEED in $(ls ${CKPT}); do
    HYP1=${CKPT}/${SEED}/${CORRECT}.en-fr.fr.hyp
    HYP2=${CKPT}/${SEED}/${INCORRECT}.en-fr.fr.hyp

    cat ${HYP1} | grep "^H" | LC_ALL=C sort -V | cut -f 2 | python ./scripts/logit2ppl.py >${CKPT}/${SEED}/commute-correct.txt &
    cat ${HYP2} | grep "^H" | LC_ALL=C sort -V | cut -f 2 | python ./scripts/logit2ppl.py >${CKPT}/${SEED}/commute-incorrect.txt &
    
    wait
    
    python ./scripts/CoMMuTE/evaluate.py \
        ${CKPT}/${SEED}/commute-correct.txt \
        ${CKPT}/${SEED}/commute-incorrect.txt \
    >${CKPT}/${SEED}/commute-score.txt
done

CKPT=checkpoint/zero-shot/en-cs_fr/vanilla/transformer_tiny
for SEED in $(ls ${CKPT}); do
    HYP1=${CKPT}/${SEED}/${CORRECT}.en-fr.fr.hyp
    HYP2=${CKPT}/${SEED}/${INCORRECT}.en-fr.fr.hyp

    cat ${HYP1} | grep "^H" | LC_ALL=C sort -V | cut -f 2 | python ./scripts/logit2ppl.py >${CKPT}/${SEED}/commute-correct.txt &
    cat ${HYP2} | grep "^H" | LC_ALL=C sort -V | cut -f 2 | python ./scripts/logit2ppl.py >${CKPT}/${SEED}/commute-incorrect.txt &
    
    wait
    
    python ./scripts/CoMMuTE/evaluate.py \
        ${CKPT}/${SEED}/commute-correct.txt \
        ${CKPT}/${SEED}/commute-incorrect.txt \
    >${CKPT}/${SEED}/commute-score.txt
done