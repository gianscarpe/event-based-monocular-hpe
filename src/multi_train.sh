s#!/bin/bash

DATASET=$1
BATCH_SIZE=32
EXECUTE=true
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
    
    get_size $MODEL
    LATENT=$retval
    
    retval="python train.py training=margipose dataset=$DATASET training.latent_size=$LATENT training.model=$MODEL training.backbone=$DATASET/classification/${MODEL}_no_aug.pt loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3"
}


echo -e ${RED} Experiments for classification ${NC}
for MODEL in resnet50 ae_resnet34_256 ae_resnet34_512
do
    get_command_with_backbone $MODEL
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

echo -e ${RED} Experiments for classification ${NC}

for MODEL in resnet34 resnet50
do 
    
    for PRETRAINED in true false   
    do
	COMMAND="python train.py training=margipose training.model=$MODEL dataset=$DATASET training.backbone=none training.pretrained=$PRETRAINED loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=3"
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
