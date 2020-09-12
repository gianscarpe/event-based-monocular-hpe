#!/bin/bash

DATASET=$1
BATCH_SIZE=32
EXECUTE=true
BASE=$2
GPU=$3

RED='\033[0;31m'
NC='\033[0m' # No Color

function get_size(){
    result=""
    arg1=$1
    if [ "$arg1" == "resnet50" ];
    then
	retval="256"
    elif [ "$arg1" == "ae_resnet34_256" ];
    then
	retval="256"
    elif [ "$arg1" == "ae_resnet34_512" ];
    then
	retval="512"
    else
	retval="128"
    fi
}

function get_command_with_backbone(){
    local MODEL=$1
    local TYPE=$2
   
    get_size $MODEL
    LATENT=$retval
    retval="python train.py gpus=[$GPU] dataset.partition=cross-view training=margipose dataset=$DATASET training.latent_size=$LATENT training.model=$MODEL training.backbone=$DATASET/${TYPE}/${MODEL}_no_aug.pt loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3"
}

function get_command_without_backbone(){
    local MODEL=$1
    local PRETRAINED=$2
    
    get_size $MODEL
    LATENT=$retval
    retval="python train.py training=margipose dataset.partition=cross-view gpus=[$GPU] training.model=$MODEL dataset=$DATASET training.latent_size=$LATENT training.backbone=none training.pretrained=$PRETRAINED loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3"
}


echo -e ${RED} Experiments for classification ${NC}




echo -e ${RED} Experiments for ae ${NC}
TYPE=autoencoder
for MODEL in ae_resnet34_cut_256 ae_resnet34_cut_512
do
    get_command_with_backbone $MODEL $TYPE
    COMMAND=$retval
    echo -e ${RED}
    echo $COMMAND
    echo -e ${NC}
    echo
    if $EXECUTE
    then
	$COMMAND
    fi

done

for MODEL in resnet34 resnet50
do     
    for PRETRAINED in true false   
    do
	get_command_without_backbone $MODEL $PRETRAINED
	COMMAND=$retval
	echo -e ${RED}
	echo $COMMAND
	echo -e ${NC}
	echo
	if $EXECUTE
	then
	    $COMMAND
	fi

    done
done
echo -e ${RED} Experiments for classification ${NC}
TYPE=classification
for MODEL in resnet34 resnet50
do
    get_command_with_backbone $MODEL $TYPE
    COMMAND=$retval
    echo -e ${RED}
    echo $COMMAND
    echo -e ${NC}
    echo
    if $EXECUTE
    then
	$COMMAND
    fi

done
