#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID res-64 \
  -nEpochs 75 \
  -batchSize 64 \
  -LR 1e-3 \
  -netType res-64 \
  -penn
