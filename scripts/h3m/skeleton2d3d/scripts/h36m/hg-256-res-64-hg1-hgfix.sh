#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID hg-256-res-64-hg1-hgfix \
  -nEpochs 50 \
  -batchSize 16 \
  -netType hg-256-res-64 \
  -penn \
  -hg \
  -hgModel ./exp/h36m/hg-256/model_best.t7 \
  -s3Model ./exp/h36m/res-64/model_best.t7 \
  -hgFix
