#!/bin/bash -eu

HYP=$1

LANG_DET=$(dirname $HYP)/$(basename $HYP .hyp).lang.txt

if [ ! -f $LANG_DET ]; then
    cat $HYP | python ./scripts/detect_tok_lang.py >$LANG_DET
fi

echo ${LANG_DET}
