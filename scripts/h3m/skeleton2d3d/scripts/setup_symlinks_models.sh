#!/bin/bash

echo "Setting up symlinks for precomputed s2d3d models and prediction..."

dir_name=( "h36m" )

cd exp

for k in "${dir_name[@]}"; do
  if [ -L $k ]; then
    rm $k
  fi
  if [ -d $k ]; then
    echo "Failed: exp/$k already exists as a folder..."
    continue
  fi
  ln -s precomputed_s2d3d_models_prediction/$k $k
done

cd ..

echo "Done."
