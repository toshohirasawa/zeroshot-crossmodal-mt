GPU_ID=$1
WORKERS=${2:-2}

find model/zero-shot \! -type d \
| grep 'en-cs_fr' \
| grep -v 'template' \
| parallel --progress -j ${WORKERS} bash {} $GPU_ID
