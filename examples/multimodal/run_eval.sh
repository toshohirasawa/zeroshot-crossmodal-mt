#!/bin/bash -eu

find checkpoint/zero-shot/en-cs_fr -name '*.hyp' \
| grep 'en-cs_fr' \
| parallel --progress -j 4 bash scripts/eval-meteor.sh {}

find checkpoint/zero-shot/en-cs_fr -name '*.hyp' \
| grep 'en-cs_fr' \
| parallel --progress -j 4 bash scripts/eval-comet.sh {} 0
