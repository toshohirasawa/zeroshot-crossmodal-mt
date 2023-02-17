#!/bin/bash -eu
export LC_ALL=en_US.UTF_8

# dependency
echo "preparing dependencies..."
if [ ! -d ./scripts/mosesdecoder ]; then
    git clone git@github.com:moses-smt/mosesdecoder.git ./scripts/mosesdecoder
fi
MOSES="./scripts/mosesdecoder/scripts/tokenizer"
export PATH="${MOSES}:$PATH"


# multi30k dataset
if [ ! -d ./data/raw/multi30k ]; then
    git clone git@github.com:multi30k/dataset.git ./data/raw/multi30k
    
    # images for test_2017_flickr, test_2018_flickr, and mscoco
    M30K=./data/raw/multi30k/data/task1
    mkdir -p ${M30K}/images
    
    # TODO: download train, validation, test_2016 images
    gdown 1mHCPUvu3anva-m0IzOLMvUYGd21y56T2 -O ${M30K}/test_2017-flickr-images.tar.gz &
    gdown 1nbpDByqmAe2v3u7E6W31tf2HYTkL1PSt -O ${M30K}/test_2018-flickr-images.tar.gz &
    wget http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/images_mscoco.task1.tar.gz -O ${M30K}/images_mscoco.task1.tar.gz &
    wait

    # TODO: extract train, validation, test_2016 images
    tar -zxvf ${M30K}/test_2017-flickr-images.tar.gz -C ${M30K}/images/ --strip-components 1 &
    tar -zxvf ${M30K}/test_2018-flickr-images.tar.gz -C ${M30K}/images/ &
    tar -zxvf ${M30K}/images_mscoco.task1.tar.gz -C ${M30K}/images/ --strip-components 1 &
    wait

    # link flickr30k images
    F30K_IMAGES=$(realpath ${FLICKR30K_IMAGES})
    for SPLIT in train val test_2016_flickr; do
        parallel -j 16 ln -s ${F30K_IMAGES}/{} ${M30K}/images/{} <${M30K}/image_splits/${SPLIT}.txt &>/dev/null &
    done
    wait
fi

# AmbigCaps
if [ ! -d ./data/raw/AmbigCaps ]; then
    wget https://polybox.ethz.ch/index.php/s/oJtNIXmnEIkfG6x/download -O ./data/raw/AmbigCaps.zip
    unzip AmbigCaps.zip -d ./data/raw/
    unzip './data/raw/AmbigCaps/*.zip' -d ./data/raw/AmbigCaps/
fi
if [ ! -d ./data/raw/AmbigCaps/tok ]; then
    mkdir -p ./data/raw/AmbigCaps/tok
    for SPLIT in train val test; do
    for LLANG in en tr; do
        GZ_FILE=./data/raw/AmbigCaps/raw/${SPLIT}.${LLANG}.gz
        TOK_FILE=./data/raw/AmbigCaps/tok/${SPLIT}.${LLANG}
        zcat $GZ_FILE | lowercase.perl | normalize-punctuation.perl -l ${LLANG} | \
            tokenizer.perl -l ${LLANG} -threads 2 >$TOK_FILE &
    done
    done
    wait
fi

# CoMMuTE
if [ ! -d ./data/raw/CoMMuTE ]; then
    git clone https://github.com/MatthieuFP/CoMMuTE ./data/raw/CoMMuTE
    # download https://drive.google.com/drive/folders/1FrvKN1PyR7zeGLllCLp50TbM0OS8LCSc
    # and place it as CoMMuTE/images.tar.gz
    gdown 1mUhqBfW4caR4t40a6d7ixc4Yx9a3oFU1 -O ./data/raw/CoMMuTE/images.tar.gz
    tar -zxvf ./data/raw/CoMMuTE/images.tar.gz -C ./data/raw/CoMMuTE/
    # remove macos system files
    rm ./data/raw/CoMMuTE/images/._*
fi

# Images
# - move all images into data/images and leave symbolic links to the original directory
#   as docker, used to extract faster-rcnn feats, requires actual files to mount them properly
echo "preparing images..."
if [ ! -d ./data/images ]; then
    mkdir -p ./data/images

    ls ./data/raw/multi30k/data/task1/images >./data/images_multi30k.txt
    # use `cp` instead of `mv`, as the source images are symbolic links.
    # `cp` will copy the actual files of the symbolic links
    # no need to re-linking files
    cp ./data/raw/multi30k/data/task1/images/* ./data/images/

    for SPLIT in train val test; do
        ls ./data/raw/AmbigCaps/${SPLIT} | grep -v 'index.txt'>./data/images_AmbigCaps_${SPLIT}.txt
        mv ./data/raw/AmbigCaps/${SPLIT}/* ./data/images/
        # move index file back into the original directory
        mv ./data/images/index.txt ./data/raw/AmbigCaps/${SPLIT}/
        cat ./data/images_AmbigCaps_${SPLIT}.txt | parallel -j 16 ln -fs ./../../../images/{} ./data/raw/AmbigCaps/${SPLIT}/{}  &>/dev/null &
    done

    ls ./data/raw/CoMMuTE/images >./data/images_CoMMuTE.txt
    mv ./data/raw/CoMMuTE/images/* ./data/images/
    cat ./data/images_CoMMuTE.txt | parallel -j 16 ln -fs ./../../../images/{} ./data/raw/CoMMuTE/images/{}  &>/dev/null &
    
    wait

    # remove index file from AmbigCaps
    ls ./data/images >./data/images.txt
fi

# exclude gif from AmbigCaps
echo "excluding gif examples from AmbigCaps..."
if [ ! -d ./data/raw/AmbigCaps/gif-excluded ]; then
    mkdir -p ./data/raw/AmbigCaps/gif-excluded/tok
    for SPLIT in train val test; do
        mkdir -p ./data/raw/AmbigCaps/gif-excluded/$SPLIT

        if [ ! -f ./data/raw/AmbigCaps/$SPLIT/mime.txt ]; then
            cat ./data/raw/AmbigCaps/$SPLIT/index.txt | \
                xargs -I{} file -i ./data/images/{} \
                >./data/raw/AmbigCaps/$SPLIT/mime.txt
        fi

        paste ./data/raw/AmbigCaps/$SPLIT/mime.txt \
            ./data/raw/AmbigCaps/tok/${SPLIT}.en \
            ./data/raw/AmbigCaps/tok/${SPLIT}.tr | \
            grep -v 'image/gif;' >./data/raw/AmbigCaps/gif-excluded/${SPLIT}.txt
        
        cut -d':' -f1 ./data/raw/AmbigCaps/gif-excluded/${SPLIT}.txt | xargs -I{} basename {} \
            >./data/raw/AmbigCaps/gif-excluded/$SPLIT/index.txt
        cut -d$'\t' -f2 ./data/raw/AmbigCaps/gif-excluded/${SPLIT}.txt \
            >./data/raw/AmbigCaps/gif-excluded/tok/${SPLIT}.en
        cut -d$'\t' -f3 ./data/raw/AmbigCaps/gif-excluded/${SPLIT}.txt \
            >./data/raw/AmbigCaps/gif-excluded/tok/${SPLIT}.tr
    done
fi

echo "generating data..."

# en-{de,fr,cs} from multi30k
M30K=./data/raw/multi30k/data/task1
REL_M30K=./../../raw/multi30k/data/task1
BPE_MOPS=10000

for L2 in "de" "fr" "cs"; do

    [ -d ./data/en-${L2} ] && continue

    echo "-> en-${L2}"

    TOK=./data/en-${L2}/tok
    BPE=./data/en-${L2}/bpe
    IMG=./data/en-${L2}/image_splits
    CODE=${BPE}/code
    VOCAB=${BPE}/vocab
    
    mkdir -p ${TOK} ${BPE} ${IMG}

    # tok
    for SPLIT in train val test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr; do
        M30K_EN=${M30K}/tok/${SPLIT}.lc.norm.tok.en
        M30K_L2=${M30K}/tok/${SPLIT}.lc.norm.tok.${L2}

        if [ -f ${M30K_L2} ]; then
            ln -s ${REL_M30K}/tok/${SPLIT}.lc.norm.tok.en ${TOK}/${SPLIT}.en
            ln -s ${REL_M30K}/tok/${SPLIT}.lc.norm.tok.${L2} ${TOK}/${SPLIT}.${L2}
            ln -s ${REL_M30K}/image_splits/${SPLIT}.txt ${IMG}/${SPLIT}.txt
        fi
    done

    # CoMMuTE
    cat ./data/raw/CoMMuTE/en-${L2}/src.en \
    | lowercase.perl | normalize-punctuation.perl -l en | tokenizer.perl -l en -threads 2 \
    >${TOK}/test_commute.en
    cat ./data/raw/CoMMuTE/en-${L2}/correct.${L2} \
    | lowercase.perl | normalize-punctuation.perl -l ${L2} | tokenizer.perl -l ${L2} -threads 2 \
    >${TOK}/test_commute.${L2}
    ln -s ./../../raw/CoMMuTE/en-${L2}/img.order ${IMG}/test_commute.txt

    # CoMMuTE - incorrect
    cat ./data/raw/CoMMuTE/en-${L2}/src.en \
    | lowercase.perl | normalize-punctuation.perl -l en | tokenizer.perl -l en -threads 2 \
    >${TOK}/test_commute_incorrect.en
    cat ./data/raw/CoMMuTE/en-${L2}/incorrect.${L2} \
    | lowercase.perl | normalize-punctuation.perl -l ${L2} | tokenizer.perl -l ${L2} -threads 2 \
    >${TOK}/test_commute_incorrect.${L2}
    ln -s ./../../raw/CoMMuTE/en-${L2}/img.order ${IMG}/test_commute_incorrect.txt

    # bpe - joint BPE with merge operations of 10k
    subword-nmt learn-joint-bpe-and-vocab -s ${BPE_MOPS} -o ${CODE} \
        --input ${TOK}/train.en ${TOK}/train.${L2} \
        --write-vocabulary ${VOCAB}.en ${VOCAB}.${L2}
    
    for SPLIT in train val test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr test_commute test_commute_incorrect; do
        if [ -f ${TOK}/${SPLIT}.${L2} ]; then
            cat ${TOK}/${SPLIT}.en \
            | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB}.en \
            >${BPE}/${SPLIT}.en &

            cat ${TOK}/${SPLIT}.${L2} \
            | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB}.${L2} \
            >${BPE}/${SPLIT}.${L2} &
        fi
    done
    wait

    # bpe - shared bpe vocab
    cat ${BPE}/train.en ${BPE}/train.${L2} | python scripts/get_vocab.py >${VOCAB}
done

# en-tr from AmbigCaps
if [ ! -d ./data/en-tr ]; then
    L2=tr
    
    echo "-> en-${L2}"

    TOK=./data/en-${L2}/tok
    BPE=./data/en-${L2}/bpe
    IMG=./data/en-${L2}/image_splits
    CODE=${BPE}/code
    VOCAB=${BPE}/vocab
    
    mkdir -p ${TOK} ${BPE} ${IMG}

    # tok
    for SPLIT in train val test; do
        for LLANG in en ${L2}; do
            if [ ${SPLIT} == test ]; then
                TGT_NAME=test_ambigcaps
            else
                TGT_NAME=${SPLIT}
            fi
            ln -s ./../../raw/AmbigCaps/gif-excluded/tok/${SPLIT}.${LLANG} ${TOK}/${TGT_NAME}.${LLANG}
        done
        ln -s ./../../raw/AmbigCaps/gif-excluded/${SPLIT}/index.txt ${IMG}/${TGT_NAME}.txt &
    done
    wait
    
    # bpe - joint BPE with merge operations of 10k
    subword-nmt learn-joint-bpe-and-vocab -s ${BPE_MOPS} -o ${CODE} \
        --input ${TOK}/train.en ${TOK}/train.${L2} \
        --write-vocabulary ${VOCAB}.en ${VOCAB}.${L2}
    
    for SPLIT in train val test_ambigcaps; do
        if [ -f ${TOK}/${SPLIT}.${L2} ]; then
            cat ${TOK}/${SPLIT}.en \
            | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB}.en \
            >${BPE}/${SPLIT}.en &

            cat ${TOK}/${SPLIT}.${L2} \
            | subword-nmt apply-bpe -c ${CODE} --vocabulary ${VOCAB}.${L2} \
            >${BPE}/${SPLIT}.${L2} &
        fi
    done
    wait

    # bpe - shared bpe vocab
    cat ${BPE}/train.en ${BPE}/train.${L2} | python scripts/get_vocab.py >${VOCAB}
fi
