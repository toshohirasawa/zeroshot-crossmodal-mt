#!/bin/bash -eu

# args
GPU=${1:-0}
SEED=${2:-$RANDOM}
N_MAX=3
echo "Random seed: ${SEED}"

# load config
CFG_NAME=$(basename $0 .sh)
TASK_TYPE=$(basename $(dirname $(dirname $(dirname $0))))
LANG_PAIR=$(basename $(dirname $(dirname $0)))
FEAT_NAME=$(basename $(dirname $0))

L1=$(echo ${LANG_PAIR} | cut -d '-' -f 1)
L2=$(echo ${LANG_PAIR} | cut -d '-' -f 2 | cut -d '_' -f 1)
L3=$(echo ${LANG_PAIR} | cut -d '-' -f 2 | cut -d '_' -f 2)

SL=$(echo ${L1} | cut -d '_' -f 1)

DATA=data-bin/${LANG_PAIR}/
LANG_LIST=data-bin/${LANG_PAIR}/lang_list.txt
CKPT_ROOT=checkpoint/${TASK_TYPE}/${LANG_PAIR}/${FEAT_NAME}/${CFG_NAME}

mkdir -p ${CKPT_ROOT}

CKPT=${CKPT_ROOT}/${SEED}

# train
if [ ! -f ${CKPT}/averaged_model.pt ]; then
  if [ -d ${CKPT} ]; then
    echo "Resuming training at ${CKPT}."
  elif [ $(ls ${CKPT_ROOT} | wc -l) -ge ${N_MAX} ]; then
    echo "Exceeded maximum number of training. Skipped."
    exit
  fi

  mkdir -p ${CKPT}

  CUDA_VISIBLE_DEVICES=${GPU} fairseq-train ${DATA} --seed ${SEED} \
    --task translation_multi_simple_epoch \
    --sampling-method "temperature" --sampling-temperature 1.5 \
    --encoder-langtok "src" --decoder-langtok \
    --lang-dict "${LANG_LIST}" --lang-pairs "en-${L2},en-${L3}" \
    --arch transformer_small_hirasawa2023 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --patience 10 \
    --update-freq 1 --no-progress-bar --log-format json --log-interval 100 \
    --save-dir ${CKPT} --keep-last-epochs 10 \
    --find-unused-parameters \
    --tensorboard-logdir ${CKPT}/tb_dir 2>&1 | tee ${CKPT}/train.log

  python scripts/average_checkpoints.py \
          --inputs ${CKPT}/ \
          --num-epoch-checkpoints 10 \
          --output ${CKPT}/averaged_model.pt
fi

# generation
if [ -f ${CKPT}/averaged_model.pt ]; then
  for TL in ${L2} ${L3}; do
  for SPLIT in valid test test1 test2 test3 test4 test5; do
    HYP=${CKPT}/${SPLIT}.${SL}-${TL}.${TL}.hyp
    LOG=${CKPT}/${SPLIT}.${SL}-${TL}.${TL}.log

    SRC_BIN=data-bin/${L1}-${L2}_${L3}/${SPLIT}.${SL}-${TL}.${TL}.bin

    if [ -f ${SRC_BIN} ] && [ ! -f ${HYP} ]; then
      (CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate ${DATA} \
        --task translation_multi_simple_epoch --source-lang ${SL} --target-lang ${TL} \
        --encoder-langtok "src" --decoder-langtok \
        --lang-dict "${LANG_LIST}" --lang-pairs "${SL}-${L2},${SL}-${L3}" \
        --gen-subset ${SPLIT} \
        --batch-size 32 --remove-bpe \
        --path ${CKPT}/averaged_model.pt \
        --beam 5 \
      | tee ${HYP}) 3>&1 1>&2 2>&3 | tee ${LOG}
    fi
  done
  done
fi
