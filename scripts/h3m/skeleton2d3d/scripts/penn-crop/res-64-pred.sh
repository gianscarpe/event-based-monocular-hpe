#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main_pred.lua \
  -expID res-64 \
  -dataset penn-crop \
  -data ./data/penn-crop \
  -modelPath ./exp/h36m/res-64/model_best.t7
