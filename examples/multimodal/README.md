## Data setup

Request the Flickr30k images here
https://forms.illinois.edu/sec/229675


Download datasets:
```
bash ./scripts/prepare-data.sh
```

Construct splits for the task:
```
bash ./scripts/prepare-data.sh
bash ./scripts/prepare-multilingual.sh
```

## Feature extraction

ResNet features:
```
bash ./scripts/extract-resnet.sh
```

Faster R-CNN features:
```
bash ./scripts/extract-faster_rcnn.sh
```

DETR features:
```
bash ./scripts/extract-detr.sh
```

CLIP features:
```
bash ./scripts/extract-clip.sh
```

## Train models

Run experiments on all configurations:
```
GPU_ID=0
WORKERS=2
bash ./run_train.sh ${GPU_ID} ${WORKERS}
```

