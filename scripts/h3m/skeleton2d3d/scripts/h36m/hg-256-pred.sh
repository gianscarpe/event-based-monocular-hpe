#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main_pred.lua \
  -expID hg-256-pred \
  -netType hg-256 \
  -penn \
  -hg \
  -modelPath ./exp/h36m/hg-256/model_best.t7
