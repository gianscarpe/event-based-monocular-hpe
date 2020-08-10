#!/bin/bash

DATASET=$1
BATCH_SIZE=16
RED='\033[0;31m'
NC='\033[0m' # No Color

"echo -e ${RED} Experiments for classification ${NC}
for MODEL in resnet34 resnet50 	     
do
    echo -e ${RED}
    echo python train.py training=margipose dataset=$DATASET    training.model=$MODEL training.backbone=$DATASET/classification/$MODEL.pt loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3
    echo -e ${NC}
    python train.py training=margipose dataset=$DATASET    training.model=$MODEL training.backbone=$DATASET/classification/$MODEL.pt loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3
    echo 
done
echo -e ${RED} Experiments for classification ${NC}
"

for MODEL in resnet34 resnet50
do 
    
    for PRETRAINED in true false   
    do
	echo -e ${RED}
	echo 	python train.py training=margipose training.model=$MODEL dataset=$DATASET training.backbone=none training.pretrawined=$PRETRAINED loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3
	python train.py training=margipose training.model=$MODEL dataset=$DATASET	training.backbone=none training.pretrained=$PRETRAINED loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3
	echo -e ${NC}
	echo
    done
done

"
echo -e ${RED} Experiments for AE ${NC}

for MODEL in ae_resnet34_256 ae_resnet34_512
do
	echo -e ${RED}     python train.py training=margipose dataset=$DATASET training.backbone=$DATASET/autoencoder/$MODEL.pt loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3
    python train.py training=margipose dataset=$DATASET training.backbone=$DATASET/autoencoder/$MODEL.pt loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3
    echo -e ${NC}
    echo
done

"
