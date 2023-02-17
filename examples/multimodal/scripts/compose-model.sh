#!/bin/bash -eu

# remove all symbolic links in model
find model -type l -delete

for TASK_TYPE in full zero-shot; do
for L2 in de fr cs tr ja; do
for L3 in de fr cs tr ja; do
    
    [ "${L2}" == "${L3}" ] && continue

    [ ! -d ./data/en-${L2}_${L3} ] && continue

    MODEL=model/${TASK_TYPE}/en-${L2}_${L3}
    TMPLT=model/${TASK_TYPE}/template

    # text-only MT
    mkdir -p ${MODEL}/vanilla/
    ln -fs ../../../../${TMPLT}/transformer_tiny.sh ${MODEL}/vanilla/transformer_tiny.sh

    # multimodal MT
    for FEAT_TYPE in $(ls data/en-${L2}_${L3}/feats); do
        
        FEAT_TYPE=$(echo ${FEAT_TYPE} | tr '[:upper:]' '[:lower:]')

        mkdir -p ${MODEL}/${FEAT_TYPE}

        for CFG in $(ls ${TMPLT}/*.sh | grep -v 'transformer_tiny' ); do
            DST=${MODEL}/${FEAT_TYPE}/$(basename ${CFG})
            ln -fs ../../../../${CFG} ${DST}
        done

    done
done
done

    L2=cs
    L3=fr

    # probing
    for MASK_TYPE in color_masking entity_masking char_masking prog_masking_5 prog_masking_10 prog_masking_15 prog_masking_20; do
        MODEL=model/${TASK_TYPE}/en_${MASK_TYPE}-${L2}_${L3}
        mkdir -p ${MODEL}/vanilla/
        ln -fs ../../../../${TMPLT}/transformer_tiny.sh ${MODEL}/vanilla/transformer_tiny.sh

        # multimodal MT
        for FEAT_TYPE in $(ls data/en_${MASK_TYPE}-${L2}_${L3}/feats); do
            
            FEAT_TYPE=$(echo ${FEAT_TYPE} | tr '[:upper:]' '[:lower:]')

            mkdir -p ${MODEL}/${FEAT_TYPE}

            for CFG in $(ls ${TMPLT}/*.sh | grep -v 'transformer_tiny' ); do
                DST=${MODEL}/${FEAT_TYPE}/$(basename ${CFG})
                ln -fs ../../../../${CFG} ${DST}
            done

        done
    done

done
