#!/bin/bash

PYTHON=${PYTHON:-"python"}

# copy files
cp compile.sh mmdetection/
cp -r src/core/* mmdetection/mmdet/core/
cp -r src/datasets/* mmdetection/mmdet/datasets/
cp -r src/models/* mmdetection/mmdet/models/
cp -r src/ops/* mmdetection/mmdet/ops/

# compile and setup
cd mmdetection
./compile.sh
$PYTHON setup.py install --user
