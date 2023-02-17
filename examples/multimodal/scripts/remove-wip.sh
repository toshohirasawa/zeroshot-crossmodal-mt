#!/bin/bash -eu

DO_DELETION=${1:-0}

for TRAIN_LOG in $(find checkpoint -mmin +10 -name "train.log"); do

    TRAIN_STATUS=$(tail -n1 ${TRAIN_LOG})

    if [[ "${TRAIN_STATUS}" != *"done training in"* ]]; then
        
        CKPT=$(dirname ${TRAIN_LOG})
        echo "--> " ${CKPT}
        ls ${CKPT}
        
        if [ "${DO_DELETION}" = "1" ]; then
            echo "Deleting"
            rm -rf ${CKPT}
        fi
    fi

done
