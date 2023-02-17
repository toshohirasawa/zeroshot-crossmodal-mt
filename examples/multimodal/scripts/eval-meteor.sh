#!/bin/bash -eu

FILE=${1}

STATE=$(grep "METEOR = " ${FILE} | wc -l)
if [ $STATE -eq 0 ]; then
    SCORE=$(python scripts/meteor_score.py --file ${FILE} 2>/dev/null)
    echo "METEOR = ${SCORE}" 1>>${FILE}
fi
