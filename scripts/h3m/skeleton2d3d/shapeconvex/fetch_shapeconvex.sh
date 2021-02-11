#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

FILE=shapeconvex.zip
URL=https://fling.seas.upenn.edu/~xiaowz/dynamic/wordpress/my-uploads/codes/shapeconvex.zip

if [ -f $FILE ]; then
  echo "File already exists..."
  exit 0
fi

echo "Downloading MPII Human Pose images (12G)..."

wget $URL -O $FILE

echo "Unzipping..."

unzip $FILE

echo "Done."
