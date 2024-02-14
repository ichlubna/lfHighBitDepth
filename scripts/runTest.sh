#!/bin/bash

APP=./lfHighBitDepth
QUALITY=../scripts/imageQualityMetrics.sh
ASPECT=1.783
RANGE=0.05_0.0975
$APP -i data/ -o out/ -g $ASPECT -r $RANGE -t 10
$QUALITY out/render.exr 0.5_0.5-centerReference.exr
ffmpeg -i out/render.exr -i 0.5_0.5-centerReference.exr -filter_complex [0:v][1:v]blend=all_mode=difference difference.exr
