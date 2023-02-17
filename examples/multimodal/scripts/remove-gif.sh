#!/bin/bash -eu

# multi30k, 
# all images are jpeg

# AmbigCaps

if [ ! -f data/image_mimes.txt ]; then
    file -i data/images/* | tee data/image_mimes.txt
fi

if [ ! -f data/image_sources.txt ]; then
    sed -e 's/$/\tmulti30k/' data/images_multi30k.txt >data/image_sources.txt
    sed -e 's/$/\tAmbigCaps_train/' data/images_AmbigCaps_train.txt >>data/image_sources.txt
    sed -e 's/$/\tAmbigCaps_val/' data/images_AmbigCaps_val.txt >>data/image_sources.txt
    sed -e 's/$/\tAmbigCaps_test/' data/images_AmbigCaps_test.txt >>data/image_sources.txt
    sed -e 's/$/\tCoMMuTE/' data/images_CoMMuTE.txt >>data/image_sources.txt
fi