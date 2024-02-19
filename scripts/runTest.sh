#!/bin/bash
# Parameters: inputDataDir referenceFile aspect range
APP=../build/lfHighBitDepth
QUALITY=./imageQualityMetrics.sh
LPIPS_DIR=~/Workspace/PerceptualSimilarity/
DISTS_DIR=~/Workspace/DISTS/DISTS_pytorch/
OUT_DIR=../build/out
ASPECT=$3
RANGE=$4
$APP -i $1 -o $OUT_DIR/ -g $ASPECT -r $RANGE -t 1
METRICS=$($QUALITY $OUT_DIR/render.exr $2)
TEMP=$(mktemp -d)
ffmpeg -i $OUT_DIR/render.exr -vf scale=3840x2160 -pix_fmt rgb48be $TEMP/render4K.png
ffmpeg -i $2 -vf scale=3840x2160 -pix_fmt rgb48be $TEMP/reference4K.png
cd $LPIPS_DIR
LPIPS=$(python lpips_2imgs.py -p0 $TEMP/render4K.png -p1 $TEMP/reference4K.png --use_gpu)
cd -
LPIPS=$(echo "$LPIPS" | tail -n1 | cut -c 10-)
ffmpeg -i $OUT_DIR/render.exr -pix_fmt rgb48be $TEMP/render.png
ffmpeg -i $2 -pix_fmt rgb48be $TEMP/reference.png
cd $DISTS_DIR
DISTS=$(python DISTS_pt.py --dist $TEMP/render.png --ref $TEMP/reference.png)
cd -
DISTS=$(echo "$DISTS" | tail -n1)
ffmpeg -i $OUT_DIR/render.exr -i $2 -filter_complex [0:v][1:v]blend=all_mode=difference $OUT_DIR/difference.exr -y
echo $METRICS $LPIPS $DISTS
rm -rf $TEMP
