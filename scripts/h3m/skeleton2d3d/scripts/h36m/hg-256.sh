#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID hg-256 \
  -nEpochs 20 \
  -batchSize 6 \
  -LR 2.5e-4 \
  -netType hg-256 \
  -penn \
  -hg \
  -hgModel ./pose-hg-train/exp/penn_action_cropped/hg-256-ft/best_model.t7
