#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )/exp"
mkdir -p $DIR && cd $DIR

FILE=precomputed_s2d3d_models_prediction.tar.gz
ID=15RObC0S1WQXEz_ArVwDWbe9VtJB_G2zU

if [ -f $FILE ]; then
  echo "File already exists..."
  exit 0
fi

echo "Downloading precomputed s2d3d models (108M)..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=$ID" -O $FILE && rm -rf /tmp/cookies.txt

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
