#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main_pred.lua \
  -expID hg-256-res-64-hg0-hgfix \
  -dataset penn-crop \
  -data ./data/penn-crop \
  -hg \
  -modelPath ./exp/h36m/hg-256-res-64-hg0-hgfix/model_best.t7
