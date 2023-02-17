#!/bin/bash -eu

ROOT_DIR=${1:-checkpoint}

if [ "${ROOT_DIR}" == "checkpoint" ]; then
    OUTFILE=results.txt
else
    OUTFILE=results-$(basename ${ROOT_DIR}).txt
fi

echo "ROOT_DIR: ${ROOT_DIR}"
echo "OUT: ${OUTFILE}"

rm -f ${OUTFILE}

for HYP in $(find ${ROOT_DIR} -name "*.hyp"); do
for METRICS in BLEU4 METEOR COMET; do
    SCORE=$(grep ${METRICS} ${HYP} | tail -n1)
    if [ ! -z "${SCORE}" ]; then
        echo ${HYP} ${SCORE} >>${OUTFILE}
    fi
done
done
