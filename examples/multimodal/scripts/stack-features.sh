#!/bin/bash -eu

FEAT_ROOT=./data/feats

for L2 in de fr cs tr; do
    
    for IM_LIST in $(ls ./data/en-${L2}/image_splits/*); do

        # avoid duplicated npy files with same content
        FEAT_ID=($(md5sum ${IM_LIST}))

        for FEAT_NAME in $(ls ${FEAT_ROOT}); do
            echo "--> ${L2} $(basename ${IM_LIST}) ${FEAT_NAME}"

            FEAT_FILE=./data/en-${L2}/feats/${FEAT_NAME}/$(basename ${IM_LIST} .txt).npy
            FEAT_TARGET=${FEAT_ROOT}/${FEAT_NAME}/stack/${FEAT_ID}.npy

            [ -f ${FEAT_FILE} ] && continue
            
            mkdir -p $(dirname ${FEAT_FILE})
            mkdir -p $(dirname ${FEAT_TARGET})

            if [ ! -f ${FEAT_TARGET} ]; then
                # determine feat-type
                if [[ ${FEAT_NAME} == clip_* ]]; then
                    FEAT_TYPE='clip'
                elif [[ ${FEAT_NAME} == detr_* ]]; then
                    FEAT_TYPE='detr'
                elif [[ ${FEAT_NAME} == faster_rcnn ]]; then
                    FEAT_TYPE='faster_rcnn'
                else
                    FEAT_TYPE='resnet'
                fi
                
                python ./scripts/stack_features.py \
                    --feat-dir ${FEAT_ROOT}/${FEAT_NAME}/raw \
                    --image-list ${IM_LIST} \
                    --output ${FEAT_TARGET} \
                    --feat-type ${FEAT_TYPE}
            fi
            
            ln -s ../../../feats/${FEAT_NAME}/stack/${FEAT_ID}.npy ${FEAT_FILE}
        done
    done
done


for L2 in de fr cs tr ja; do
for L3 in de fr cs tr ja; do
    [ ! -d ./data/en-${L2}_${L3} ] && continue
    [ "${L2}" == "${L3}" ] && continue

    for IM_LIST in $(ls ./data/en-${L2}_${L3}/image_splits/*); do

        # avoid duplicated npy files with same content
        FEAT_ID=($(md5sum ${IM_LIST}))

        for FEAT_NAME in $(ls ${FEAT_ROOT}); do
            echo "--> ${L2}_${L3} $(basename ${IM_LIST}) ${FEAT_NAME}"

            FEAT_FILE=./data/en-${L2}_${L3}/feats/${FEAT_NAME}/$(basename ${IM_LIST} .txt).npy
            FEAT_TARGET=${FEAT_ROOT}/${FEAT_NAME}/stack/${FEAT_ID}.npy

            [ -f ${FEAT_FILE} ] && continue
            
            mkdir -p $(dirname ${FEAT_FILE})
            mkdir -p $(dirname ${FEAT_TARGET})

            if [ ! -f ${FEAT_TARGET} ]; then
                # determine feat-type
                if [[ ${FEAT_NAME} == clip_* ]]; then
                    FEAT_TYPE='clip'
                elif [[ ${FEAT_NAME} == detr_* ]]; then
                    FEAT_TYPE='detr'
                elif [[ ${FEAT_NAME} == faster_rcnn ]]; then
                    FEAT_TYPE='faster_rcnn'
                else
                    FEAT_TYPE='resnet'
                fi
                
                python ./scripts/stack_features.py \
                    --feat-dir ${FEAT_ROOT}/${FEAT_NAME}/raw \
                    --image-list ${IM_LIST} \
                    --output ${FEAT_TARGET} \
                    --feat-type ${FEAT_TYPE}
            fi
            
            ln -s ../../../feats/${FEAT_NAME}/stack/${FEAT_ID}.npy ${FEAT_FILE}
        done
    done
done
done